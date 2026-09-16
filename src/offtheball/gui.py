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
from .metrics import metric_log

BG = "#101821"
PANEL = "#192532"
TEXT = "#e7edf2"
MUTED = "#a5b6c6"
ACCENT = "#8ee5b0"
TABLET_DEFAULT_URL = "http://10.50.75.89:8080"


class App:
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
        self.photo = None
        self.display = None
        self.closing = False
        self.status = tk.StringVar(value="영상을 선택하면 시작할 수 있습니다.")
        self.detail = tk.StringVar(value="팀 분류 · 선수 추적 · 0.2")
        self.file_label = tk.StringVar(value="선택한 영상 없음")
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(60, self.poll)

    def _build(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("맑은 고딕", 10), padding=9,
                        background="#263849", foreground=TEXT)
        style.map("TButton", background=[("active", "#385269"), ("disabled", "#202832")])
        header = tk.Frame(self.root, bg=BG, padx=24, pady=8)
        header.pack(fill="x")
        tk.Label(header, text="OFF THE BALL", font=("Segoe UI", 19, "bold"),
                 fg=ACCENT, bg=BG).pack(anchor="w")
        tk.Label(header, text="축구 영상의 움직임을 읽는 첫 단계",
                 fg=MUTED, bg=BG, font=("맑은 고딕", 10)).pack(anchor="w")
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20)
        side = tk.Frame(body, bg=PANEL, width=265, padx=17, pady=18)
        side.pack(side="left", fill="y", padx=(0, 14))
        side.pack_propagate(False)
        tk.Label(side, text="01   영상 입력", bg=PANEL, fg=ACCENT,
                 font=("맑은 고딕", 11, "bold")).pack(anchor="w", pady=(0, 8))
        self.open_button = ttk.Button(side, text="영상 파일 선택", command=self.open_video)
        self.open_button.pack(fill="x")
        tk.Label(side, textvariable=self.file_label, bg=PANEL, fg=MUTED, wraplength=225,
                 justify="left", font=("맑은 고딕", 9)).pack(fill="x", pady=12)
        self.run_button = ttk.Button(side, text="영상 분석 시작", command=self.start_video)
        self.run_button.pack(fill="x", pady=4)
        self.screen_button = ttk.Button(side, text="화면 영역 분석", command=self.start_screen)
        self.screen_button.pack(fill="x", pady=4)
        self.tablet_button = ttk.Button(side, text="태블릿 카메라 연결", command=self.start_tablet)
        self.tablet_button.pack(fill="x", pady=4)
        self.stop_button = ttk.Button(side, text="분석 중지", command=self.stop.set, state="disabled")
        self.stop_button.pack(fill="x", pady=4)
        tk.Label(side, text="02   정지 장면 실험", bg=PANEL, fg=ACCENT,
                 font=("맑은 고딕", 11, "bold")).pack(anchor="w", pady=(18, 7))
        self.cal_button = ttk.Button(side, text="현재 장면 거리 보정", command=self.calibrate)
        self.cal_button.pack(fill="x")
        self.board_button = ttk.Button(side, text="전술 보드 표시", command=self.toggle_board)
        self.board_button.pack(fill="x", pady=4)
        self.formation_button = ttk.Button(side, text="기준 배치 설정", command=self.open_formation)
        self.formation_button.pack(fill="x", pady=4)
        tk.Label(side, text="고정 카메라의 경기장 지점을 한 번 보정하면\n영상 전체에서 미터 좌표를 계산합니다.",
                 bg=PANEL, fg=MUTED, justify="left", font=("맑은 고딕", 9)).pack(anchor="w", pady=10)
        ttk.Button(side, text="결과 폴더 열기", command=self.open_outputs).pack(fill="x", pady=(12, 0))
        tk.Label(side, text="현재 기능\n사람 탐지 · 임시 번호 추적\n팀 A/B 색상 · 화면상 궤적\n정지 장면의 수동 거리 측정\n\nA 파랑 · B 빨강 · ? 미확인",
                 bg=PANEL, fg=MUTED, justify="left",
                 font=("맑은 고딕", 9)).pack(anchor="w", pady=12)
        view = tk.Frame(body, bg=PANEL)
        view.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(view, bg="#0a1118", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        board_panel = tk.Frame(view, bg="#0d3b24", width=320, height=420)
        board_panel.pack(side="right", fill="y")
        board_panel.pack_propagate(False)
        tk.Label(board_panel, text="전술 보드", bg="#0d3b24", fg="white",
                 font=("맑은 고딕", 11, "bold")).pack(pady=(10, 4))
        self.board = tk.Canvas(board_panel, bg="#11833a", width=300, height=225,
                               highlightthickness=0)
        self.board.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(board_panel, text="실시간 분석 로그", bg="#0d3b24", fg="white",
                 font=("맑은 고딕", 9, "bold")).pack(anchor="w", padx=10)
        self.board_log = tk.Text(board_panel, height=7, bg="#092b1a", fg="#d8f3df",
                                 relief="flat", state="disabled", wrap="word",
                                 font=("Consolas", 8))
        self.board_log.pack(fill="both", expand=True, padx=10, pady=(3, 8))
        self.board_panel = board_panel
        self.board_panel.pack_forget()
        self.canvas.bind("<Configure>", lambda e: self.paint())
        self.canvas.create_text(300, 200, text="경기 영상을 선택하세요",
                                fill=MUTED, font=("맑은 고딕", 18), tags="empty")
        tk.Label(view, textvariable=self.detail, bg=PANEL, fg=MUTED,
                 anchor="w", padx=14, pady=12, font=("맑은 고딕", 10)).pack(fill="x")
        tk.Label(self.root, textvariable=self.status, bg=BG, fg=TEXT, anchor="w",
                 padx=24, pady=16, font=("맑은 고딕", 10)).pack(fill="x")

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def controls(self, running):
        for button in (self.open_button, self.run_button, self.screen_button,
                       self.tablet_button, self.cal_button, self.board_button,
                       self.formation_button):
            button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def toggle_board(self):
        if self.board_panel.winfo_manager():
            self.board_panel.pack_forget()
            self.board_button.configure(text="전술 보드 표시")
        else:
            self.board_panel.pack(side="right", anchor="ne", padx=8, pady=8)
            self.board_button.configure(text="전술 보드 숨기기")
            self.paint_board()

    def open_formation(self):
        if self.busy():
            return
        FormationWindow(self.root, on_saved=self._formation_saved)

    def _formation_saved(self, payload):
        self.formation = payload
        self.status.set("기준 배치를 저장했습니다. 선수 간 기준 거리와 팀 폭을 계산할 수 있습니다.")

    def open_video(self):
        if self.busy():
            return
        path = filedialog.askopenfilename(title="분석할 영상 선택",
                    filetypes=[("영상", "*.mp4 *.avi *.mkv *.mov *.webm"), ("모든 파일", "*.*")])
        if not path:
            return
        try:
            with VideoSource(path) as source:
                frame = source.read()
                if frame is None:
                    raise ValueError("읽을 수 있는 프레임이 없습니다.")
            self.clear_preview_queue()
            self.path = Path(path)
            self.last_raw = frame.pixels
            self.display = frame.pixels
            self.file_label.set(self.path.name)
            self.status.set("영상 분석을 시작하거나 현재 장면을 수동 보정하세요.")
            self.detail.set(f"{frame.width} × {frame.height} · 첫 프레임")
            self.paint()
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

    def start_video(self):
        if self.path is None:
            messagebox.showinfo("영상 선택", "먼저 영상 파일을 선택하세요.")
            return
        path = self.path
        def run():
            detector = self.get_detector()
            output, summary = analyze_video(
                path, detector, stop=self.stop,
                calibration=self.calibration,
                on_frame=lambda rendered, row, raw: self.publish(rendered, raw, row))
            self.events.put(("result", (output, summary)))
        self.launch_worker(run)

    def start_screen(self):
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
        def run():
            detector = self.get_detector()
            session = AnalysisSession(detector, calibration=self.calibration)
            self.events.put(("status", "화면 분석 중 · 고정 카메라 보정 적용" if self.calibration else
                             "화면 분석 중 · 거리 미보정 · 중지를 누르면 종료합니다."))
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
                    detections = session.process(frame)
                    rendered = annotate(frame.pixels, detections)
                    self.publish(rendered, frame.pixels,
                                 {"screen": True, "detections": [d.to_dict() for d in detections],
                                  "processing_s": time.monotonic() - tick,
                                  "source_time_s": frame.timestamp_s})
                    self.stop.wait(.01)
        self.launch_worker(run)

    def start_tablet(self):
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

        def run():
            detector = self.get_detector()
            session = AnalysisSession(detector, calibration=self.calibration)
            self.events.put(("status", f"태블릿 카메라 연결 중 · {redact_url(source.url)}"))
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
                        detections = session.process(frame)
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
            self.display = rendered
            if raw is not None:
                self.last_raw = raw
            else:
                self.last_raw = None
            self.last_info = info
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

    def paint_board(self):
        if not hasattr(self, "board"):
            return
        self.board.delete("all")
        bw = max(self.board.winfo_width(), 180)
        bh = max(self.board.winfo_height(), 240)
        pad = 18
        left, top, right, bottom = pad, pad, bw - pad, bh - pad
        self.board.create_rectangle(left, top, right, bottom, outline="white", width=2)
        mid = (left + right) / 2
        self.board.create_line(mid, top, mid, bottom, fill="white")
        self.board.create_oval(mid - (right-left)*.115, (top+bottom)/2 - (bottom-top)*.115,
                               mid + (right-left)*.115, (top+bottom)/2 + (bottom-top)*.115,
                               outline="white")
        for x in (left, right):
            self.board.create_rectangle(x, (top+bottom)/2-(bottom-top)*.20,
                                        x + (right-left)*(.13 if x == left else -.13),
                                        (top+bottom)/2+(bottom-top)*.20, outline="white")
        detections = getattr(self, "last_info", {}).get("detections", [])
        info = getattr(self, "last_info", {})
        if hasattr(self, "board_log") and detections:
            teams = {key: sum(item.get("team") == key for item in detections) for key in ("A", "B")}
            calibrated = sum(bool(item.get("field_xy")) for item in detections)
            stamp = info.get("source_time_s", info.get("received_time_s", 0.0))
            line = (f"{float(stamp):6.1f}s  선수 {len(detections):2d}  "
                    f"A {teams['A']:2d} / B {teams['B']:2d} / 좌표 {calibrated:2d}\n")
            self.board_log.configure(state="normal")
            self.board_log.insert("end", line)
            if calibrated >= 2:
                self.board_log.insert("end", f"         {metric_log(detections)}\n")
            self.board_log.see("end")
            # Keep the live panel bounded during long sessions.
            if int(self.board_log.index("end-1c").split(".")[0]) > 80:
                self.board_log.delete("1.0", "20.0")
            self.board_log.configure(state="disabled")
        if not detections:
            self.board.create_text((left+right)/2, bottom+20, text="선수 위치 대기 중",
                                   fill="white", font=("맑은 고딕", 9))
            return
        # Use calibrated metres when available; otherwise keep a clearly labelled
        # normalized screen estimate so the board remains useful during setup.
        calibrated = any(item.get("field_xy") for item in detections)
        for item in detections:
            field = item.get("field_xy")
            if field and len(field) == 2:
                x, y = float(field[0]), float(field[1])
                px = left + (x / 105.0) * (right-left)
                py = bottom - (y / 68.0) * (bottom-top)
            else:
                box = item.get("xyxy") or [0, 0, 0, 0]
                px = left + ((box[0]+box[2]) / 2.0 / max(1, self.last_raw.shape[1])) * (right-left)
                py = top + (box[3] / max(1, self.last_raw.shape[0])) * (bottom-top)
            px, py = max(left+4, min(right-4, px)), max(top+4, min(bottom-4, py))
            team = item.get("team")
            color = "#4aa3ff" if team == "A" else "#ff5b5b" if team == "B" else "#b7c2c9"
            self.board.create_oval(px-8, py-8, px+8, py+8, fill=color, outline="white")
            label = str(item.get("track_id") or "?")
            self.board.create_text(px, py, text=label, fill="white", font=("Segoe UI", 8, "bold"))
        self.board.create_text((left+right)/2, bottom+20,
                               text="미터 기준" if calibrated else "화면 기준 · 보정 필요",
                               fill="white", font=("맑은 고딕", 9))

    def calibrate(self):
        if self.busy():
            return
        raw = self.last_raw
        if raw is None:
            messagebox.showinfo("장면 선택", "먼저 영상을 열거나 화면 분석을 중지하세요.")
            return
        if is_black_frame(raw):
            messagebox.showinfo("입력 확인", "검은 화면은 경기장 거리 보정에 사용할 수 없습니다.")
            return
        CalibrationWindow(self.root, raw.copy(), on_saved=self._set_calibration)

    def _set_calibration(self, calibration):
        self.calibration = calibration
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
        data = {"version": 1, "field_size_m": {"length": 105.0, "width": 68.0},
                "players": points, "team_width_m": max(p[0] for p in points)-min(p[0] for p in points)}
        path = ROOT / "formation.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if callable(self.on_saved):
            self.on_saved(data)
        self.window.destroy()


class CalibrationWindow:
    def __init__(self, parent, pixels, on_saved=None):
        from .calibration import ManualCalibration
        self.Calibration = ManualCalibration
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
