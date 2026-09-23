"""Seekable analysis preview. Video decoder and tracker belong to one worker."""
import json
import queue
import time
import threading
from tkinter import messagebox

from .inputs import VideoSource
from .analysis import AnalysisSession
from .detection import annotate
from .pipeline import create_run_directory
from .quality import is_black_frame


class PlaybackMixin:
    def init_playback(self):
        self.duration = 0.0
        self.playhead = 0.0
        self.playback_commands = queue.Queue()
        self.playback_paused = threading.Event()
        self.preview_only = False
        self.file_playback = False
        self.pitch_polygon = None
        self.pitch_size = None
        self.requested_seek = 0

    def seek_video(self, event=None):
        self.scrubbing = False
        if self.path is None or (self.busy() and not self.file_playback):
            return
        self.playhead = float(self.seek_position.get())
        self.clear_preview_queue()
        if self.busy():
            self.requested_seek += 1
            self.playback_commands.put(("seek", (self.playhead, self.requested_seek)))
        else:
            # A stopped video can be inspected at any time without loading AI.
            with VideoSource(self.path) as source:
                source.seek(self.playhead)
                frame = source.read()
            if frame is not None:
                self.display = self.last_raw = frame.pixels
                self.last_info = {"source_time_s": frame.timestamp_s, "detections": []}
                self.playhead = frame.timestamp_s
                self.paint()
                self.refresh_playhead()

    def refresh_playhead(self):
        stamp = getattr(self, "last_info", {}).get("source_time_s", self.playhead)
        self.playhead = float(stamp)
        if not getattr(self, "scrubbing", False):
            self.seek_position.set(self.playhead)
        def clock(value):
            return f"{int(value)//60:02d}:{int(value)%60:02d}"
        self.time_label.set(f"{clock(self.playhead)} / {clock(self.duration)}")

    def toggle_pause(self):
        if not self.busy():
            self.start_video()
        elif self.file_playback:
            if self.playback_paused.is_set():
                self.playback_paused.clear()
                self.pause_button.configure(text="Ⅱ")
            else:
                self.playback_paused.set()
                self.pause_button.configure(text="▶")

    def start_video(self):
        if self.busy():
            return
        if self.path is None:
            messagebox.showinfo("영상 선택", "먼저 영상 파일을 선택하세요.")
            return
        if self.pitch_polygon is None:
            self.select_pitch(on_ready=self.start_video)
            return
        self.file_playback = True
        self.requested_seek = 0
        self.playback_paused.clear()
        while not self.playback_commands.empty():
            self.playback_commands.get_nowait()
        path, start = self.path, self.playhead
        polygon = self.pitch_polygon
        calibration = self.calibration
        def run():
            output = create_run_directory()
            self.events.put(("preview_output", output))
            try:
                session = AnalysisSession(self.get_detector(), calibration=calibration,
                                          pitch_polygon=polygon)
                with VideoSource(path) as source, (output / "frames.jsonl").open("w", encoding="utf-8") as records:
                    source.seek(start)
                    epoch = 0
                    force_frame = True
                    while not self.stop.is_set():
                        target = None
                        while True:
                            try:
                                command, target = self.playback_commands.get_nowait()
                            except queue.Empty:
                                break
                        if target is not None:
                            source.seek(target[0])
                            session.reset()
                            epoch = target[1]
                            force_frame = True
                        if self.playback_paused.is_set() and not force_frame:
                            self.stop.wait(.03)
                            continue
                        tick = time.monotonic()
                        frame = source.read()
                        if frame is None:
                            break
                        detections = session.process(frame, black=is_black_frame(frame.pixels))
                        row = {"schema_version": 2,
                               "source_time_s": frame.timestamp_s, "frame_index": frame.index,
                               "scene_id": session.scene_id, "seek_epoch": epoch,
                               "detections": [d.to_dict() for d in detections],
                               "calibration_status": session.calibration_diagnostics()["status"],
                               "calibration": session.calibration_diagnostics(),
                               "processing_s": time.monotonic() - tick}
                        records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        self.publish(annotate(frame.pixels, detections), frame.pixels, row)
                        force_frame = False
                        self.stop.wait(max(0, 1 / source.fps - (time.monotonic() - tick)))
                self.events.put(("status", "분석 종료 · 선택한 구간의 좌표 기록을 결과 폴더에 저장했습니다."))
            finally:
                self.file_playback = False
                self.playback_paused.clear()
        self.launch_worker(run)
