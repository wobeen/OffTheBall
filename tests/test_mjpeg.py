"""Synthetic local MJPEG checks; no internet or camera is contacted."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import cv2
import numpy as np
import pytest

from offtheball.inputs import MJPEGError, MJPEGSource, resolve_mjpeg_url


def jpeg(value: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((24, 32, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


def multipart(frames: list[bytes]) -> bytes:
    parts = []
    for frame in frames:
        parts.append(b"--test\r\nContent-Type: image/jpeg\r\nContent-Length: "
                     + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
    return b"".join(parts) + b"--test--\r\n"


class StreamHandler(BaseHTTPRequestHandler):
    payload = b""
    keep_open = False
    chunk_size = 0

    def do_GET(self):  # noqa: N802
        if self.path != "/video":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=test")
        self.send_header("Connection", "close" if not self.keep_open else "keep-alive")
        self.end_headers()
        if self.chunk_size:
            for start in range(0, len(self.payload), self.chunk_size):
                self.wfile.write(self.payload[start:start + self.chunk_size])
                self.wfile.flush()
                time.sleep(.001)
        else:
            self.wfile.write(self.payload)
            self.wfile.flush()
        if self.keep_open:
            while True:
                time.sleep(.05)

    def log_message(self, *_args):
        pass


def serve(payload: bytes, *, keep_open: bool = False, chunk_size: int = 0):
    handler = type("TestStreamHandler", (StreamHandler,),
                   {"payload": payload, "keep_open": keep_open, "chunk_size": chunk_size})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_resolve_page_url_to_standard_video_endpoint():
    assert resolve_mjpeg_url("http://10.50.75.89:8080") == "http://10.50.75.89:8080/video"
    assert resolve_mjpeg_url("http://10.50.75.89:8080/") == "http://10.50.75.89:8080/video"
    assert resolve_mjpeg_url("http://127.0.0.1:1/custom.mjpg") == "http://127.0.0.1:1/custom.mjpg"
    with pytest.raises(ValueError):
        resolve_mjpeg_url("file:///tmp/camera")


def test_fragmented_mjpeg_and_latest_only_buffer():
    frames = [jpeg(value) for value in (20, 80, 160)]
    payload = multipart(frames)
    server, _ = serve(payload, chunk_size=7)
    try:
        with MJPEGSource(f"http://127.0.0.1:{server.server_port}", read_timeout=.3) as source:
            source._thread.join(timeout=5)
            assert not source._thread.is_alive(), "finite test stream must finish"
            frame = source.read(timeout=1)
            assert frame is not None
            # The reader may have consumed all three before the consumer runs;
            # it must never retain an unbounded queue of stale frames.
            assert source._frames.qsize() <= 1
            assert frame.width == 32 and frame.height == 24
            assert frame.index >= 2
            assert float(frame.pixels.mean()) > 120
    finally:
        server.shutdown()
        server.server_close()


def test_disconnect_raises_after_last_frame_and_close_stops_promptly():
    payload = multipart([jpeg(100)])
    server, _ = serve(payload)
    try:
        with MJPEGSource(f"http://127.0.0.1:{server.server_port}", read_timeout=.2) as source:
            assert source.read(timeout=1) is not None
            with pytest.raises(MJPEGError, match="스트림이 종료"):
                for _ in range(5):
                    source.read(timeout=.3)
    finally:
        server.shutdown()
        server.server_close()

    server, _ = serve(b"", keep_open=True)
    started = time.monotonic()
    source = MJPEGSource(f"http://127.0.0.1:{server.server_port}", read_timeout=5)
    with source:
        time.sleep(.05)
    assert time.monotonic() - started < 1.5
    server.shutdown()
    server.server_close()


def test_invalid_url_and_redacted_error():
    with pytest.raises(ValueError):
        MJPEGSource("not a URL")
    server, _ = serve(b"", keep_open=False)
    try:
        source = MJPEGSource(f"http://user:secret@127.0.0.1:{server.server_port}")
        with source:
            with pytest.raises(MJPEGError) as error:
                source.read(timeout=1)
        assert "secret" not in str(error.value)
    finally:
        server.shutdown()
        server.server_close()


def test_source_cannot_be_reopened_after_close():
    server, _ = serve(multipart([jpeg(40)]))
    source = MJPEGSource(f"http://127.0.0.1:{server.server_port}")
    with source:
        assert source.read(timeout=1) is not None
    with pytest.raises(RuntimeError, match="cannot be reopened"):
        source.__enter__()
    server.shutdown()
    server.server_close()
