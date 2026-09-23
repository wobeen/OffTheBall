"""Korean desktop UI. Workers publish only the latest preview to Tk."""
from __future__ import annotations
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from . import ROOT
from .inputs import MJPEGError, MJPEGSource, VideoSource, ScreenSource, redact_url
from .detection import PersonDetector, annotate
from .analysis import AnalysisSession
from .pipeline import analyze_video, create_run_directory
from .quality import is_black_frame
from .metrics import metric_log, team_metrics, formation_deviation, formation_message
from .events import EventTimeline, EVENT_LABELS, summarize_event
from .playback import PlaybackMixin
from .dashboard import DashboardMixin

BG = "#101821"
PANEL = "#192532"
TEXT = "#e7edf2"
MUTED = "#a5b6c6"
ACCENT = "#8ee5b0"
TABLET_DEFAULT_URL = "http://10.50.75.89:8080"


class App(DashboardMixin, PlaybackMixin):
    def __init__(self, root):
        self.root = root
        root.title("OffTheBall · 오프더볼")
        root.geometry("1220x800")
        root.minsize(980, 650)
        root.configure(bg=BG)
        self.path = None
        self.last_raw = None
        self.last_output = ROOT / "outputs"
        self.worker = None
        self.stop = threading.Event()
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.detector = None
        self.calibration = None
        self.event_timeline = EventTimeline()
        self.analysis_rows = []
        self.photo = None
        self.display = None
        self.closing = False
        self.status = tk.StringVar(value="영상을 선택하면 시작할 수 있습니다.")
        self.detail = tk.StringVar(value="팀 분류 · 선수 추적 · 0.2")
        self.file_label = tk.StringVar(value="선택한 영상 없음")
        self.init_playback()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(60, self.poll)

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def open_formation(self):
        FormationWindow(self.root, on_saved=self._formation_saved)

    def show_help(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("OffTheBall 사용 방법")
        dialog.geometry("760x650")
        dialog.minsize(620, 480)
        dialog.configure(bg=BG)
        text = tk.Text(dialog, wrap="word", bg="#111c28", fg=TEXT,
                       insertbackground=TEXT, relief="flat", padx=18, pady=16,
                       font=("맑은 고딕", 10), spacing1=3, spacing3=7)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True, padx=(12, 0), pady=12)
        text.insert("end", """빠른 시작

1. 영상 열기로 경기 영상을 선택합니다.
2. 기준 보정을 눌러 영상 좌표를 105 × 68m 경기장 좌표에 연결합니다.
3. 분석 시작을 누릅니다. 처음에는 그라운드 영역을 둘러 지정합니다.
4. 분석 중 영상의 선수 표시와 전술 보드의 점이 함께 움직이는지 확인합니다.

기준 보정

전술 보드에 선수를 표시하고 거리·간격을 계산하기 위한 필수 단계입니다. 경기장 선이 넓게 보이는 장면에서 실제 위치를 아는 지점 4개 이상을 선택하고 같은 방향의 X,Y 미터 좌표를 입력하세요. 빠른 동작 확인에는 자동 기준점 찾기를 사용할 수 있지만, 녹색 영역 전체를 경기장 전체로 가정하므로 실제 거리 분석에는 부정확할 수 있습니다.

보정 창에서는 기준점을 입력한 뒤 보정 계산, 보정·장면 저장 순서로 누릅니다. 파일 분석 중 기준 보정을 누르면 현재 프레임에서 자동으로 일시정지합니다. 저장한 보정은 현재 분석에도 적용되며 ▶ 버튼으로 다시 재생할 수 있습니다. 화면·태블릿 입력은 움직이는 실시간 프레임이므로 먼저 중지한 뒤 보정하세요.

그라운드 영역

관중과 벤치 인원이 추적기에 들어오지 않도록 영상에서 실제 경기 영역만 지정합니다. 그라운드 영역 선택과 기준 보정은 서로 다른 기능이며 둘 다 필요합니다.

기준 배치

우리 팀이 기대하는 11명의 위치를 경기장 위에서 드래그해 저장합니다. 분석 중에도 열고 수정할 수 있습니다. 현재는 팀 A의 실제 배치와 비교해 팀 폭과 위치 편차를 계산합니다.

이벤트와 브리핑

분석 중 중요한 장면에서 이벤트 태그를 남기면 브리핑에서 전후 구간의 팀 폭과 배치 편차를 확인할 수 있습니다.

전술 보드에 선수가 보이지 않을 때

• '기준 보정 후 선수 위치가 표시됩니다': 기준 보정을 저장하지 않은 상태입니다.
• '경기장 좌표로 계산할 수 없습니다': 선수 발 위치가 보정 기준점의 유효 영역 밖일 수 있습니다.
• 영상을 다시 열면 이전 보정은 초기화됩니다. 같은 영상을 연 상태에서 보정 후 분석하세요.
• 방송 카메라가 팬·줌·장면 전환되면 한 장면의 수동 보정은 정확하지 않을 수 있습니다.

번호와 분석 결과의 의미

화면의 번호는 실제 등번호나 선수 신원이 아니라 현재 장면 안의 임시 추적 ID입니다. 팀 A/B도 유니폼 색상으로 추정하며 새 입력에서는 대응이 바뀔 수 있습니다.
""")
        text.configure(state="disabled")
        ttk.Button(dialog, text="닫기", command=dialog.destroy,
                   style="Dashboard.TButton").pack(pady=(0, 12))

    def _formation_saved(self, payload):
        self.formation = payload
        self.status.set("기준 배치를 저장했습니다. 선수 간 기준 거리와 팀 폭을 계산할 수 있습니다.")

    def tag_event(self):
        stamp = float(getattr(self, "last_info", {}).get("source_time_s", 0.0))
        dialog = tk.Toplevel(self.root)
        dialog.title("이벤트 태그")
        dialog.configure(bg=BG)
        tk.Label(dialog, text=f"현재 시점 {stamp:.1f}초", bg=BG, fg=TEXT).pack(padx=16, pady=10)
        label = tk.StringVar(value=EVENT_LABELS[0])
        ttk.Combobox(dialog, textvariable=label, values=EVENT_LABELS, state="readonly").pack(padx=16)
        note = tk.Entry(dialog, width=36)
        note.pack(padx=16, pady=10)
        def save():
            self.event_timeline.add(stamp, label.get(), note.get())
            self.event_timeline.save(ROOT / "events.json")
            self.status.set(f"{stamp:.1f}초 {label.get()} 이벤트를 저장했습니다.")
            dialog.destroy()
        ttk.Button(dialog, text="저장", command=save).pack(pady=(0, 12))

    def show_briefing(self):
        if not self.event_timeline.events:
            messagebox.showinfo("브리핑", "먼저 영상 분석 중 이벤트를 하나 이상 태그하세요.")
            return
        reports = [summarize_event(event, self.analysis_rows, formation=getattr(self, "formation", None))
                   for event in self.event_timeline.events]
        dialog = tk.Toplevel(self.root)
        dialog.title("전술 브리핑")
        text = tk.Text(dialog, width=72, height=20, wrap="word")
        text.pack(padx=12, pady=12)
        for report in reports:
            text.insert("end", f"[{report['timestamp_s']:.1f}초] {report['label']}\n")
            text.insert("end", f"분석 샘플 {report['samples']}개\n")
            for team, values in report["teams"].items():
                if values["mean_width_m"] is not None:
                    text.insert("end", f"팀 {team} 평균 폭 {values['mean_width_m']:.1f}m\n")
            if report.get("formation"):
                text.insert("end", f"기준 배치 최대 편차 {report['formation']['max_m']:.1f}m\n")
            text.insert("end", "\n")
        text.configure(state="disabled")

    def open_video(self):
        if self.busy():
            return
        path = filedialog.askopenfilename(title="분석할 영상 선택",
                    filetypes=[("영상", "*.mp4 *.avi *.mkv *.mov *.webm"), ("모든 파일", "*.*")])
        if not path:
            return
        try:
            with VideoSource(path) as source:
                self.duration = source.duration_s
                frame = source.read()
                if frame is None:
                    raise ValueError("읽을 수 있는 프레임이 없습니다.")
            self.clear_preview_queue()
            self.path = Path(path)
            self.input_key = ("file", str(self.path.resolve()))
            self.pitch_polygon = None
            self.calibration = None
            self.event_timeline.clear()
            self.analysis_rows.clear()
            self.playhead = 0.0
            self.last_info = {}
            self.seek_scale.configure(to=max(.01, self.duration))
            self.seek_position.set(0)
            self.last_raw = frame.pixels
            self.display = frame.pixels
            self.file_label.set(self.path.name)
            self.status.set("영상 분석을 시작하거나 현재 장면을 수동 보정하세요.")
            self.detail.set(f"{frame.width} × {frame.height} · 첫 프레임")
            self.paint()
            self.refresh_playhead()
        except Exception as exc:
            messagebox.showerror("영상을 열 수 없습니다", str(exc))

    def get_detector(self):
        if self.detector is None:
            self.events.put(("status", "탐지 모델을 준비하고 있습니다. 첫 실행은 조금 걸릴 수 있습니다."))
            self.detector = PersonDetector()
        return self.detector

    def publish(self, rendered, raw, info):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait((rendered, raw, info))

    def clear_preview_queue(self):
        try:
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass

    def clear_display(self):
        """Remove both the cached image and all canvas items."""
        self.clear_preview_queue()
        self.display = None
        self.last_raw = None
        self.last_info = {}
        self.canvas.delete("all")

    def launch_worker(self, target):
        if self.busy():
            return
        self.clear_display()
        self.analysis_rows.clear()
        self.stop.clear()
        self.controls(True)
        self.status.set("분석을 준비하고 있습니다.")
        def run():
            try:
                target()
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                self.events.put(("finished", None))
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def export_video(self):
        if self.busy():
            return
        if self.path is None:
            messagebox.showinfo("영상 선택", "먼저 영상 파일을 선택하세요.")
            return
        if self.pitch_polygon is None:
            self.select_pitch(on_ready=self.export_video)
            return
        path = self.path
        def run():
            detector = self.get_detector()
            output, summary = analyze_video(
                path, detector, stop=self.stop,
                calibration=self.calibration, pitch_polygon=self.pitch_polygon,
                on_frame=lambda rendered, row, raw: self.publish(rendered, raw, row))
            self.events.put(("result", (output, summary)))
        self.launch_worker(run)

    def start_screen(self):
        if self.busy():
            return
        text = simpledialog.askstring("화면 영역",
            "분석할 화면 영역을 입력하세요: 왼쪽, 위쪽, 너비, 높이\n"
            "예: 0, 0, 960, 540\n분석 창이 이 영역과 겹치면 화면이 반복 캡처됩니다.",
            parent=self.root)
        if text is None:
            return
        try:
            values = [int(v.strip()) for v in text.split(",")]
            if len(values) != 4:
                raise ValueError()
            region = dict(zip(("left", "top", "width", "height"), values))
            ScreenSource(region)
        except (TypeError, ValueError):
            messagebox.showerror("영역 확인", "정수 4개와 양수인 너비·높이를 입력하세요.")
            return
        key = ("screen", tuple(values))
        if getattr(self, "input_key", None) != key:
            self.pitch_polygon = None
            self.calibration = None
        self.input_key = key
        self.path = None
        self.duration = self.playhead = 0.0
        self.seek_position.set(0)
        self.time_label.set("LIVE")
        self.file_playback = False
        def run():
            detector = self.get_detector()
            session = AnalysisSession(detector, calibration=self.calibration, pitch_polygon=self.pitch_polygon)
            self.events.put(("status", "화면 분석 중 · 고정 카메라 보정 적용" if self.calibration else
                             "화면 분석 중 · 거리 미보정 · 중지를 누르면 종료합니다."))
            if self.pitch_polygon is None:
                self.events.put(("status", "화면 미리보기 · 중지 → 더보기 → 그라운드 영역 선택 후 같은 화면에 다시 연결하세요."))
            with ScreenSource(region) as source:
                while not self.stop.is_set():
                    tick = time.monotonic()
                    frame = source.read()
                    # A blank capture must not appear to be a successful analysis.
                    if is_black_frame(frame.pixels):
                        session.process(frame, black=True)
                        self.publish(frame.pixels, frame.pixels,
                                     {"screen": True, "input_status": "black_frame", "message": "검은 화면 · 입력을 확인하세요."})
                        self.stop.wait(.2)
                        continue
                    detections = session.process(frame) if self.pitch_polygon is not None else []
                    rendered = annotate(frame.pixels, detections)
                    self.publish(rendered, frame.pixels,
                                 {"screen": True, "detections": [d.to_dict() for d in detections],
                                  "processing_s": time.monotonic() - tick,
                                  "source_time_s": frame.timestamp_s})
                    self.stop.wait(.01)
        self.launch_worker(run)

    def start_tablet(self):
        if self.busy():
            return
        text = simpledialog.askstring(
            "태블릿 카메라",
            "IP Webcam 주소를 입력하세요. 기본 주소를 그대로 사용하면 /video 스트림으로 연결합니다.",
            initialvalue=TABLET_DEFAULT_URL,
            parent=self.root,
        )
        if text is None:
            return
        try:
            source = MJPEGSource(text)
        except (TypeError, ValueError) as exc:
            messagebox.showerror("카메라 주소 확인", str(exc), parent=self.root)
            return

        key = ("tablet", source.url)
        if getattr(self, "input_key", None) != key:
            self.pitch_polygon = None
            self.calibration = None
        self.input_key = key
        self.path = None
        self.duration = self.playhead = 0.0
        self.seek_position.set(0)
        self.time_label.set("LIVE")
        self.file_playback = False
        def run():
            detector = self.get_detector()
            session = AnalysisSession(detector, calibration=self.calibration, pitch_polygon=self.pitch_polygon)
            self.events.put(("status", f"태블릿 카메라 연결 중 · {redact_url(source.url)}"))
            if self.pitch_polygon is None:
                self.events.put(("status", "태블릿 미리보기 · 중지 → 더보기 → 그라운드 영역 선택 후 같은 주소에 다시 연결하세요."))
            try:
                with source:
                    got_frame = False
                    frame_stall_timeout = max(5.0, source.connect_timeout + source.read_timeout + 2.0)
                    last_valid_frame_at = time.monotonic()
                    while not self.stop.is_set():
                        frame = source.read(timeout=.5)
                        if frame is None:
                            if time.monotonic() - last_valid_frame_at >= frame_stall_timeout:
                                if got_frame:
                                    raise MJPEGError("태블릿 카메라에서 새 JPEG 프레임이 멈췄습니다.")
                                raise MJPEGError("태블릿 카메라에서 유효한 JPEG 프레임을 받지 못했습니다.")
                            continue
                        last_valid_frame_at = time.monotonic()
                        if not got_frame:
                            got_frame = True
                            self.events.put(("status", f"태블릿 카메라 분석 중 · {redact_url(source.url)}"))
                        # A blank input must not be reported as a successful
                        # person analysis.
                        if is_black_frame(frame.pixels):
                            session.process(frame, black=True)
                            self.publish(frame.pixels, frame.pixels,
                                         {"tablet": True, "input_status": "black_frame",
                                          "message": "검은 화면 · 태블릿 촬영 방향을 확인하세요."})
                            continue
                        tick = time.monotonic()
                        detections = session.process(frame) if self.pitch_polygon is not None else []
                        rendered = annotate(frame.pixels, detections)
                        self.publish(rendered, frame.pixels,
                                     {"tablet": True,
                                      "detections": [d.to_dict() for d in detections],
                                      "processing_s": time.monotonic() - tick,
                                      # This is local receive/decode elapsed time,
                                      # not a camera capture timestamp or latency.
                                      "received_time_s": frame.timestamp_s})
            except MJPEGError as exc:
                self.events.put(("input_lost", str(exc)))
                raise
            finally:
                session.reset()
                source.close()
                if self.stop.is_set():
                    self.events.put(("status", "태블릿 카메라 분석을 중지했습니다."))

        self.launch_worker(run)

    def poll(self):
        if not self.root.winfo_exists():
            return
        try:
            rendered, raw, info = self.frames.get_nowait()
            if self.file_playback and info.get("seek_epoch", 0) < self.requested_seek:
                raise queue.Empty
            self.display = rendered
            if raw is not None:
                self.last_raw = raw
            else:
                self.last_raw = None
            self.last_info = info
            if self.file_playback or "source_time_s" in info:
                self.refresh_playhead()
            if info.get("detections"):
                self.analysis_rows.append({"source_time_s": info.get("source_time_s", info.get("received_time_s", 0.0)),
                                           "detections": info.get("detections", [])})
            detections = info.get('detections', [])
            teams = {label: sum(1 for item in detections if item.get('team') == label)
                     for label in ('A', 'B')}
            count_text = f"A {teams['A']} · B {teams['B']} · 미확인 {sum(1 for item in detections if item.get('team') not in ('A', 'B'))}"
            if info.get("tablet"):
                self.detail.set((info.get("message") or
                    f"태블릿 수신 프레임 · 사람 {len(detections)}명 · {count_text} · "
                    f"처리 {info.get('processing_s', 0)*1000:.0f}ms · 화면상 궤적"))
            else:
                self.detail.set(("검은 화면 · 선수 유무를 판단할 수 없습니다." if info.get("input_status") == "black_frame" else info.get("message")) or
                    f"사람 {len(detections)}명 · {count_text} · "
                    f"{info.get('source_time_s', 0):.1f}초 · "
                    f"처리 {info.get('processing_s', 0)*1000:.0f}ms · 화면상 궤적")
            self.paint()
        except queue.Empty:
            pass
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "status":
                    self.status.set(payload)
                elif event == "preview_output":
                    self.last_output = payload
                elif event == "result":
                    self.last_output, summary = payload
                    word = "완료" if summary["status"] == "completed" else "중지"
                    self.status.set(f"분석 {word} · {summary['frames']}프레임 · 검은 화면 {summary.get('black_frames', 0)}프레임 · 결과 저장")
                elif event == "error":
                    self.status.set("분석을 완료하지 못했습니다.")
                    if not self.closing:
                        messagebox.showerror("분석 오류", payload)
                elif event == "input_lost":
                    # A lost/reconnecting network stream must never leave an
                    # old frame or old detections looking like live output.
                    self.clear_display()
                    self.detail.set(payload)
                elif event == "finished":
                    self.controls(False)
                    if self.status.get().startswith("화면 분석"):
                        self.status.set("화면 분석을 중지했습니다. 현재 장면을 보정할 수 있습니다.")
        except queue.Empty:
            pass
        if self.closing and not self.busy():
            self.root.destroy()
            return
        self.root.after(60, self.poll)

    def paint(self):
        if self.display is None:
            return
        w, h = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        image = Image.fromarray(cv2.cvtColor(self.display, cv2.COLOR_BGR2RGB))
        image.thumbnail((w, h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(w//2, h//2, image=self.photo, anchor="center")
        self.paint_board()

    def select_pitch(self, on_ready=None):
        if self.busy():
            return
        if self.last_raw is None:
            messagebox.showinfo("그라운드 영역", "영상을 열거나 실시간 입력을 중지한 뒤 영역을 지정하세요.")
            return
        from .pitch_editor import PitchEditor
        size = (self.last_raw.shape[1], self.last_raw.shape[0])
        def save(polygon):
            from .pitch import PitchPolygon
            self.pitch_polygon = PitchPolygon.from_points(polygon, width=size[0], height=size[1])
            self.pitch_size = size
            self.status.set("그라운드 영역 저장 · 이 영역 안에 발이 있는 사람만 추적합니다.")
            if on_ready:
                on_ready()
        PitchEditor(self.root, self.last_raw.copy(), save)

    def calibrate(self):
        if self.busy() and not self.file_playback:
            messagebox.showinfo(
                "분석 중 기준 보정",
                "화면·태블릿·전체 영상 저장 중에는 현재 입력을 먼저 중지한 뒤 기준 보정을 눌러주세요.",
            )
            return
        if self.busy() and self.file_playback:
            self.playback_paused.set()
            self.pause_button.configure(text="▶")
            self.status.set("현재 프레임에서 일시정지했습니다. 기준 보정 후 ▶로 계속할 수 있습니다.")
        raw = self.last_raw
        if raw is None:
            messagebox.showinfo("장면 선택", "먼저 영상을 열거나 화면 분석을 중지하세요.")
            return
        if is_black_frame(raw):
            messagebox.showinfo("입력 확인", "검은 화면은 경기장 거리 보정에 사용할 수 없습니다.")
            return
        CalibrationWindow(self.root, raw.copy(), on_saved=self._set_calibration)

    def _set_calibration(self, calibration):
        from .calibration import AutoPitchCalibration
        self.calibration = AutoPitchCalibration(calibration)
        if self.busy() and self.file_playback:
            self.playback_commands.put(("calibration", self.calibration))
            self.status.set("기준 보정을 현재 분석에 적용했습니다. ▶를 누르면 전술 보드가 갱신됩니다.")
        else:
            self.status.set("고정 카메라 보정이 준비되었습니다. 다음 분석부터 미터 좌표를 기록합니다.")
        self.detail.set(f"경기장 보정 활성 · {calibration.image_size[0]} × {calibration.image_size[1]} · 미터 좌표 사용 가능")

    def open_outputs(self):
        self.last_output.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.last_output))

    def close(self):
        self.closing = True
        self.stop.set()
        self.controls(True)
        self.status.set("결과를 정리하고 종료하고 있습니다.")
        if not self.busy():
            self.root.destroy()


class FormationWindow:
    """Drag eleven reference players on a 105 x 68 metre pitch."""
    def __init__(self, parent, on_saved=None):
        self.on_saved = on_saved
        self.window = tk.Toplevel(parent)
        self.window.title("기준 전술 배치 · 105 × 68m")
        self.window.configure(bg=BG)
        self.w, self.h = 760, 500
        self.canvas = tk.Canvas(self.window, width=self.w, height=self.h,
                                bg="#11833a", highlightthickness=0)
        self.canvas.pack(padx=14, pady=14)
        self.positions = [[18 + i * 6, 20 + (i % 3) * 14] for i in range(11)]
        self.dragging = None
        self._draw()
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "dragging", None))
        bar = tk.Frame(self.window, bg=BG)
        bar.pack(fill="x", padx=14, pady=(0, 12))
        ttk.Button(bar, text="기준 배치 저장", command=self._save).pack(side="right")

    def _xy(self, event):
        return max(0, min(105, event.x / self.w * 105)), max(0, min(68, (self.h-event.y) / self.h * 68))

    def _draw(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(8, 8, self.w-8, self.h-8, outline="white", width=2)
        self.canvas.create_line(self.w/2, 8, self.w/2, self.h-8, fill="white")
        self.canvas.create_oval(self.w*.5-55, self.h*.5-55, self.w*.5+55, self.h*.5+55, outline="white")
        for i, (x, y) in enumerate(self.positions, 1):
            px, py = x / 105 * self.w, self.h - y / 68 * self.h
            self.canvas.create_oval(px-14, py-14, px+14, py+14, fill="#4aa3ff", outline="white", tags=f"p{i}")
            self.canvas.create_text(px, py, text=str(i), fill="white", font=("Segoe UI", 10, "bold"))

    def _press(self, event):
        x, y = self._xy(event)
        distances = [((px-x)**2 + (py-y)**2, i) for i, (px, py) in enumerate(self.positions)]
        distance, index = min(distances)
        if distance < 80:
            self.dragging = index

    def _drag(self, event):
        if self.dragging is not None:
            self.positions[self.dragging] = list(self._xy(event))
            self._draw()

    def _save(self):
        points = [list(map(float, point)) for point in self.positions]
        data = {"version": 1, "team": "A", "field_size_m": {"length": 105.0, "width": 68.0},
                "players": points, "team_width_m": max(p[0] for p in points)-min(p[0] for p in points)}
        path = ROOT / "formation.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if callable(self.on_saved):
            self.on_saved(data)
        self.window.destroy()


class CalibrationWindow:
    def __init__(self, parent, pixels, on_saved=None):
        from .calibration import ManualCalibration, suggest_pitch_corners
        self.Calibration = ManualCalibration
        self.suggest_pitch_corners = suggest_pitch_corners
        self.pixels = pixels
        self.h, self.w = pixels.shape[:2]
        self.frame_id = uuid.uuid4().hex
        self.points = []
        self.field = []
        self.measure = []
        self.calibration = None
        self.on_saved = on_saved
        self.window = tk.Toplevel(parent)
        self.window.title("정지 장면 거리 실험")
        self.window.configure(bg=BG)
        self.note = tk.StringVar(value="알려진 경기장 지점 4개 이상을 클릭하고 각 지점의 실제 좌표(m)를 입력하세요.")
        tk.Label(self.window, textvariable=self.note, bg=BG, fg=TEXT, wraplength=940,
                 font=("맑은 고딕", 10), pady=12).pack(fill="x")
        self.scale = min(1000/self.w, 620/self.h, 1)
        self.canvas = tk.Canvas(self.window, width=round(self.w*self.scale),
                                height=round(self.h*self.scale), highlightthickness=0)
        self.canvas.pack()
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                    .resize((round(self.w*self.scale), round(self.h*self.scale))))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.bind("<Button-1>", self.click)
        bar = tk.Frame(self.window, bg=BG, pady=10)
        bar.pack(fill="x")
        ttk.Button(bar, text="보정 계산", command=self.fit).pack(side="left", padx=8)
        ttk.Button(bar, text="자동 기준점 찾기", command=self.auto_points).pack(side="left", padx=8)
        ttk.Button(bar, text="처음부터", command=self.reset).pack(side="left", padx=8)
        ttk.Button(bar, text="보정·장면 저장", command=self.save).pack(side="left", padx=8)
        tk.Label(self.window, text="고정 카메라라면 이 보정을 영상 전체에 재사용할 수 있습니다. 팬·줌·위치 변화 후에는 다시 보정하세요.",
                 bg=BG, fg=MUTED, pady=10).pack()

    def click(self, event):
        point = [event.x/self.scale, event.y/self.scale]
        if self.calibration is None:
            value = simpledialog.askstring("경기장 좌표",
                    "클릭한 지점의 실제 X,Y 좌표를 미터로 입력하세요. 예: 0, 0\n"
                    "모든 지점은 같은 원점과 방향을 사용해야 합니다.", parent=self.window)
            if value is None:
                return
            try:
                xy = [float(v.strip()) for v in value.split(",")]
                if len(xy) != 2 or not np.isfinite(xy).all():
                    raise ValueError()
            except ValueError:
                messagebox.showerror("좌표 확인", "유한한 숫자 두 개를 입력하세요.", parent=self.window)
                return
            self.points.append(point)
            self.field.append(xy)
            self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4,
                                    fill=ACCENT, tags="mark")
            self.canvas.create_text(event.x+9, event.y-10, text=str(len(self.points)),
                                    fill="white", tags="mark")
            self.note.set(f"기준점 {len(self.points)}개 · 4개 이상을 넓게 지정한 뒤 보정 계산을 누르세요.")
        else:
            xy = self.calibration.project([point], image_size=(self.w, self.h), frame_id=self.frame_id)[0]
            if not np.isfinite(xy).all():
                self.note.set("기준점이 둘러싼 범위 밖입니다. 이 지점의 거리는 계산하지 않습니다.")
                self.measure.clear()
                self.canvas.delete("measure")
                return
            if len(self.measure) == 2:
                self.measure.clear()
                self.canvas.delete("measure")
            self.measure.append((point, xy))
            self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4,
                                    fill="#ffc96d", tags="measure")
            if len(self.measure) == 2:
                a, b = self.measure
                distance = np.linalg.norm(a[1]-b[1])
                self.canvas.create_line(a[0][0]*self.scale, a[0][1]*self.scale,
                                        b[0][0]*self.scale, b[0][1]*self.scale,
                                        fill="#ffc96d", width=2, tags="measure")
                self.note.set(f"두 지점의 추정 거리: {distance:.2f} m · 기준점 정확도에 따라 오차가 있습니다.")
            else:
                self.note.set("두 번째 측정 지점을 클릭하세요.")

    def fit(self):
        try:
            self.calibration = self.Calibration.fit(self.points, self.field,
                                image_size=(self.w, self.h), frame_id=self.frame_id)
            self.note.set("보정 계산 완료 · 기준점 안쪽의 두 지점을 클릭하면 거리를 측정합니다.")
        except ValueError as exc:
            messagebox.showerror("보정 불가", str(exc), parent=self.window)

    def auto_points(self):
        points = self.suggest_pitch_corners(self.pixels)
        if points is None:
            messagebox.showinfo("자동 기준점", "경기장 영역을 찾지 못했습니다. 수동으로 기준점을 지정하세요.", parent=self.window)
            return
        self.points = points
        self.field = [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]]
        self.canvas.delete("mark")
        for index, point in enumerate(points, 1):
            px, py = point[0] * self.scale, point[1] * self.scale
            self.canvas.create_oval(px-4, py-4, px+4, py+4, fill=ACCENT, tags="mark")
            self.canvas.create_text(px+9, py-10, text=str(index), fill="white", tags="mark")
        self.calibration = None
        self.note.set("자동 기준점 4개를 제안했습니다. 화면을 확인한 뒤 보정 계산을 누르세요.")

    def reset(self):
        self.points.clear()
        self.field.clear()
        self.measure.clear()
        self.calibration = None
        self.canvas.delete("mark")
        self.canvas.delete("measure")
        self.note.set("알려진 경기장 지점 4개 이상을 클릭하고 실제 좌표를 입력하세요.")

    def save(self):
        if self.calibration is None:
            messagebox.showinfo("보정 필요", "먼저 기준점을 입력하고 보정을 계산하세요.", parent=self.window)
            return
        folder = create_run_directory()
        self.calibration.save(folder / "calibration.json")
        if callable(self.on_saved):
            self.on_saved(self.calibration)
        ok, data = cv2.imencode(".png", self.pixels)
        if not ok:
            raise RuntimeError("장면 이미지를 저장할 수 없습니다.")
        data.tofile(str(folder / "frame.png"))
        self.note.set(f"보정과 원본 장면을 저장했습니다: {folder.name}")


def launch():
    root = tk.Tk()
    App(root)
    root.mainloop()
