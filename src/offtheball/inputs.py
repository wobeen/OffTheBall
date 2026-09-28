"""Explicit video-file, desktop-region, and local MJPEG frame inputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import queue
import threading
import time
from collections.abc import Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests


@dataclass(frozen=True)
class Frame:
    pixels: np.ndarray
    index: int
    timestamp_s: float
    captured_at: float

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError("Frames must contain three-channel BGR pixels")
        if self.pixels.dtype != np.uint8 or min(self.pixels.shape[:2]) <= 0:
            raise ValueError("Frames must contain nonempty uint8 pixels")

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    @property
    def height(self) -> int:
        return self.pixels.shape[0]


class VideoSource:
    """Read a local video; timestamps are relative to the start of the file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self._capture = None
        self._index = 0
        self._last_timestamp = -1.0

    def __enter__(self) -> VideoSource:
        if self._capture is not None:
            raise RuntimeError("Video source is already open")
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open video: {self.path}")
        raw_fps = capture.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else 30.0
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if min(self.width, self.height) <= 0:
            capture.release()
            raise ValueError("Video has invalid dimensions")
        self._capture = capture
        self._index = 0
        self._last_timestamp = -1.0
        return self

    def read(self) -> Frame | None:
        if self._capture is None:
            raise RuntimeError("Open VideoSource with a with statement before reading")
        ok, pixels = self._capture.read()
        if not ok:
            return None
        captured_at = time.monotonic()
        timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= self._last_timestamp:
            timestamp = max(self._index / self.fps, self._last_timestamp + 1.0 / self.fps)
        frame = Frame(pixels, self._index, timestamp, captured_at)
        self.width, self.height = frame.width, frame.height
        self._last_timestamp = timestamp
        self._index += 1
        return frame

    @property
    def duration_s(self):
        if self._capture is None:
            return 0.0
        return max(0.0, self._capture.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps)

    def seek(self, seconds):
        if self._capture is None:
            raise RuntimeError("Video source is not open")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Seek time must be finite and nonnegative")
        index = min(round(seconds * self.fps), max(0, int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError("영상의 선택한 시점으로 이동하지 못했습니다.")
        self._index = index
        self._last_timestamp = -1.0

    def __iter__(self) -> Iterator[Frame]:
        while (frame := self.read()) is not None:
            yield frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _create_screen_grabber():
    import mss
    return mss.mss()


class ScreenSource:
    """Capture only a caller-selected desktop rectangle, in BGR format.

    Construct, open and read on the same thread (required by the capture backend).
    """

    def __init__(self, region: Mapping[str, int]) -> None:
        required = {"left", "top", "width", "height"}
        if not isinstance(region, Mapping) or set(region) != required:
            raise ValueError("Choose an explicit region: left, top, width, height")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in region.values()):
            raise ValueError("Screen region values must be integers")
        if region["width"] <= 0 or region["height"] <= 0:
            raise ValueError("Screen region dimensions must be positive")
        self.region = dict(region)
        self.width = region["width"]
        self.height = region["height"]
        self._grabber = None
        self._index = 0
        self._started = 0.0

    def __enter__(self) -> ScreenSource:
        if self._grabber is not None:
            raise RuntimeError("Screen source is already open")
        grabber = _create_screen_grabber()
        try:
            bounds = grabber.monitors[0]
            region = self.region
            if (region["left"] < bounds["left"] or region["top"] < bounds["top"]
                or region["left"] + region["width"] > bounds["left"] + bounds["width"]
                or region["top"] + region["height"] > bounds["top"] + bounds["height"]):
                raise ValueError("Selected region is outside the desktop bounds")
        except Exception:
            grabber.close()
            raise
        self._grabber = grabber
        self._index = 0
        self._started = time.monotonic()
        return self

    def read(self) -> Frame:
        if self._grabber is None:
            raise RuntimeError("Open ScreenSource with a with statement before reading")
        pixels = np.asarray(self._grabber.grab(self.region))
        captured_at = time.monotonic()
        if pixels.shape != (self.height, self.width, 4):
            raise ValueError("Screen capture dimensions changed; select the region again")
        frame = Frame(pixels[:, :, :3].copy(), self._index,
                      captured_at - self._started, captured_at)
        self._index += 1
        return frame

    def __iter__(self) -> Iterator[Frame]:
        while True:
            yield self.read()

    def close(self) -> None:
        if self._grabber is not None:
            self._grabber.close()
            self._grabber = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class MJPEGError(RuntimeError):
    """A safe, user-facing error from a network MJPEG input."""


def redact_url(url: str) -> str:
    """Return a URL suitable for an error message without credentials."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or "호스트 없음"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path or "/", "", ""))
    except (TypeError, ValueError):
        return "(잘못된 URL)"


def resolve_mjpeg_url(url: str) -> str:
    """Resolve an IP Webcam page URL to its standard ``/video`` endpoint.

    IP Webcam advertises the multipart MJPEG endpoint at ``/video``.  A
    direct stream URL is kept intact so other local MJPEG cameras can also be
    used.  This does not open a listening socket or expose the camera.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("카메라 URL을 입력하세요.")
    value = url.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("카메라 URL은 http:// 또는 https:// 주소여야 합니다.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("카메라 URL의 포트를 확인하세요.") from exc
    path = parsed.path or "/"
    if path in {"", "/", "/greet.html"}:
        path = "/video"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


class _JPEGParser:
    """Bounded JPEG marker parser for multipart streams."""

    MAX_BUFFER = 12 * 1024 * 1024
    MAX_FRAME = 10 * 1024 * 1024

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        if chunk:
            self.buffer.extend(chunk)
        if len(self.buffer) > self.MAX_BUFFER:
            raise MJPEGError("태블릿 카메라 스트림이 너무 커서 중단했습니다.")
        frames: list[bytes] = []
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start < 0:
                # Keep a possible marker split across two network chunks.
                if len(self.buffer) > 1:
                    del self.buffer[:-1]
                break
            if start:
                del self.buffer[:start]
            end = self.buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(self.buffer) > self.MAX_FRAME:
                    raise MJPEGError("태블릿 카메라의 JPEG 프레임이 너무 큽니다.")
                break
            end += 2
            if end > self.MAX_FRAME:
                raise MJPEGError("태블릿 카메라의 JPEG 프레임이 너무 큽니다.")
            frames.append(bytes(self.buffer[:end]))
            del self.buffer[:end]
        return frames


class MJPEGSource:
    """Read a local HTTP multipart MJPEG stream with a latest-frame buffer.

    Network reading happens on a daemon thread so ``close()`` can promptly
    release the response and unblock a worker.  The queue is size one and
    drops old frames whenever analysis falls behind.
    """

    def __init__(self, url: str, *, connect_timeout: float = 3.0,
                 read_timeout: float = 2.0) -> None:
        self.requested_url = url.strip() if isinstance(url, str) else url
        self.url = resolve_mjpeg_url(self.requested_url)
        self.connect_timeout = max(float(connect_timeout), 0.1)
        self.read_timeout = max(float(read_timeout), 0.1)
        self.width = 0
        self.height = 0
        self._frames: queue.Queue[Frame] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._response = None
        self._session = None
        self._closed = False
        self._error: Exception | None = None
        self._index = 0
        self._started = 0.0
        self._lock = threading.Lock()

    def __enter__(self) -> "MJPEGSource":
        if self._thread is not None:
            raise RuntimeError("MJPEG source is already open")
        if self._closed:
            raise RuntimeError("MJPEG source cannot be reopened after close")
        self._stop.clear()
        self._ready.clear()
        self._error = None
        self._index = 0
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._reader, name="offtheball-mjpeg", daemon=True)
        self._thread.start()
        return self

    def _set_error(self, error: Exception) -> None:
        with self._lock:
            if self._error is None and not self._stop.is_set():
                self._error = error
        self._ready.set()

    def _reader(self) -> None:
        parser = _JPEGParser()
        try:
            # A desktop proxy must not redirect this explicitly supplied LAN
            # camera request to the internet or a dead proxy listener.
            session = requests.Session()
            session.trust_env = False
            self._session = session
            response = session.get(self.url, stream=True,
                                   timeout=(self.connect_timeout, self.read_timeout))
            with self._lock:
                self._response = response
            if response.status_code < 200 or response.status_code >= 300:
                raise MJPEGError(f"태블릿 카메라가 HTTP {response.status_code}를 반환했습니다.")
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "multipart" not in content_type and "jpeg" not in content_type:
                raise MJPEGError("태블릿 카메라가 MJPEG 스트림을 반환하지 않았습니다.")
            self._ready.set()
            for chunk in response.iter_content(chunk_size=16384):
                if self._stop.is_set():
                    break
                for encoded in parser.feed(chunk):
                    try:
                        pixels = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                    except cv2.error:
                        # A malformed JPEG must not kill the reader thread or
                        # leave the UI waiting forever for a valid frame.
                        continue
                    if pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3:
                        continue
                    captured_at = time.monotonic()
                    frame = Frame(pixels, self._index, captured_at - self._started, captured_at)
                    self._index += 1
                    self.width, self.height = frame.width, frame.height
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frames.put_nowait(frame)
                    except queue.Full:
                        pass
                    self._ready.set()
            if not self._stop.is_set():
                raise MJPEGError("태블릿 카메라 스트림이 종료되었습니다.")
        except requests.RequestException as exc:
            if not self._stop.is_set():
                self._set_error(MJPEGError(f"태블릿 카메라에 연결하지 못했습니다: {redact_url(self.url)}"))
        except (MJPEGError, OSError, ValueError) as exc:
            if not self._stop.is_set():
                self._set_error(exc if isinstance(exc, MJPEGError)
                                else MJPEGError(f"태블릿 카메라 스트림을 읽지 못했습니다: {redact_url(self.url)}"))
        finally:
            with self._lock:
                response = self._response
                self._response = None
            if response is not None:
                response.close()
            session = self._session
            self._session = None
            if session is not None:
                session.close()
            self._ready.set()

    def read(self, timeout: float = 0.5) -> Frame | None:
        if self._thread is None:
            raise RuntimeError("Open MJPEGSource with a with statement before reading")
        try:
            return self._frames.get(timeout=max(float(timeout), 0.0))
        except queue.Empty:
            with self._lock:
                error = self._error
            if error is not None:
                raise error
            return None

    def __iter__(self) -> Iterator[Frame]:
        while not self._stop.is_set():
            frame = self.read()
            if frame is not None:
                yield frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._lock:
            response = self._response
            self._response = None
        if response is not None:
            # Some server/socket combinations can block while closing a
            # response.  Do that cleanup off the UI/worker stop path.
            threading.Thread(target=response.close, name="offtheball-mjpeg-close",
                             daemon=True).start()
        session = self._session
        self._session = None
        if session is not None:
            threading.Thread(target=session.close, name="offtheball-mjpeg-session-close",
                             daemon=True).start()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            # Closing a requests response normally interrupts iter_content;
            # keep the UI stop path bounded even if a server ignores it.
            thread.join(timeout=1.0)
        self._thread = None
        self._ready.clear()
        try:
            while True:
                self._frames.get_nowait()
        except queue.Empty:
            pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
