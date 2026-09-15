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
from .inputs import VideoSource, ScreenSource
from .detection import PersonDetector, annotate
from .pipeline import analyze_video, create_run_directory
from .quality import is_black_frame

BG = "#101821"
PANEL = "#192532"
TEXT = "#e7edf2"
MUTED = "#a5b6c6"
ACCENT = "#8ee5b0"


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
        self.photo = None
        self.display = None
        self.closing = False
        self.status = tk.StringVar(value="영상을 선택하면 시작할 수 있습니다.")
        self.detail = tk.StringVar(value="사람 탐지 기준 버전 · 0.1")
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
        header = tk.Frame(self.root, bg=BG, padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="OFF THE BALL", font=("Segoe UI", 23, "bold"),
                 fg=ACCENT, bg=BG).pack(anchor="w")
        tk.Label(header, text="축구 영상의 움직임을 읽는 첫 단계",
                 fg=MUTED, bg=BG, font=("맑은 고딕", 11)).pack(anchor="w")
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
        self.stop_button = ttk.Button(side, text="분석 중지", command=self.stop.set, state="disabled")
        self.stop_button.pack(fill="x", pady=4)
        tk.Label(side, text="02   정지 장면 실험", bg=PANEL, fg=ACCENT,
                 font=("맑은 고딕", 11, "bold")).pack(anchor="w", pady=(25, 9))
        self.cal_button = ttk.Button(side, text="현재 장면 거리 보정", command=self.calibrate)
        self.cal_button.pack(fill="x")
        tk.Label(side, text="알고 있는 경기장 지점으로 보정한 뒤\n두 지점의 거리를 측정합니다.",
                 bg=PANEL, fg=MUTED, justify="left", font=("맑은 고딕", 9)).pack(anchor="w", pady=10)
        ttk.Button(side, text="결과 폴더 열기", command=self.open_outputs).pack(fill="x", pady=(18, 0))
        tk.Label(side, text="현재 기능\n사람 탐지 · 결과 영상 저장\n정지 장면의 수동 거리 측정\n\n팀 구분·추적·자동 보정은\n다음 개발 단계입니다.",
                 bg=PANEL, fg=MUTED, justify="left",
                 font=("맑은 고딕", 9)).pack(anchor="w", pady=20)
        view = tk.Frame(body, bg=PANEL)
        view.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(view, bg="#0a1118", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
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
        for button in (self.open_button, self.run_button, self.screen_button, self.cal_button):
            button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

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

    def launch_worker(self, target):
        if self.busy():
            return
        self.clear_preview_queue()
        self.last_raw = None
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
            self.events.put(("status", "화면 분석 중 · 거리 미보정 · 중지를 누르면 종료합니다."))
            with ScreenSource(region) as source:
                while not self.stop.is_set():
                    tick = time.monotonic()
                    frame = source.read()
                    # A blank capture must not appear to be a successful analysis.
                    if is_black_frame(frame.pixels):
                        self.publish(frame.pixels, frame.pixels,
                                     {"screen": True, "input_status": "black_frame", "message": "검은 화면 · 입력을 확인하세요."})
                        self.stop.wait(.2)
                        continue
                    detections = detector.detect(frame.pixels)
                    rendered = annotate(frame.pixels, detections)
                    self.publish(rendered, frame.pixels,
                                 {"screen": True, "detections": [d.to_dict() for d in detections],
                                  "processing_s": time.monotonic() - tick,
                                  "source_time_s": frame.timestamp_s})
                    self.stop.wait(.01)
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
            self.detail.set(("검은 화면 · 선수 유무를 판단할 수 없습니다." if info.get("input_status") == "black_frame" else info.get("message")) or
                f"사람 {len(info.get('detections', []))}명 · "
                f"{info.get('source_time_s', 0):.1f}초 · "
                f"처리 {info.get('processing_s', 0)*1000:.0f}ms · 거리 미보정")
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
        CalibrationWindow(self.root, raw.copy())

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


class CalibrationWindow:
    def __init__(self, parent, pixels):
        from .calibration import ManualCalibration
        self.Calibration = ManualCalibration
        self.pixels = pixels
        self.h, self.w = pixels.shape[:2]
        self.frame_id = uuid.uuid4().hex
        self.points = []
        self.field = []
        self.measure = []
        self.calibration = None
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
        tk.Label(self.window, text="정지된 이 장면에만 유효합니다. 다른 시각·카메라 위치에는 적용하지 않습니다.",
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
        ok, data = cv2.imencode(".png", self.pixels)
        if not ok:
            raise RuntimeError("장면 이미지를 저장할 수 없습니다.")
        data.tofile(str(folder / "frame.png"))
        self.note.set(f"보정과 원본 장면을 저장했습니다: {folder.name}")


def launch():
    root = tk.Tk()
    App(root)
    root.mainloop()
