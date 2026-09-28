"""Compact dashboard layout for :mod:`offtheball.gui`.

The mixin deliberately owns presentation only.  ``App`` continues to own
input, analysis, playback, and event state; integrating the dashboard is a
matter of adding ``DashboardMixin`` before ``App`` in the application class.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .metrics import formation_deviation, team_metrics


BG = "#101724"
SURFACE = "#151f2d"
SURFACE_RAISED = "#1b2939"
BORDER = "#2b3c52"
TEXT = "#ecf4f7"
MUTED = "#8da4b5"
MINT = "#9bf3c5"
MINT_DARK = "#143c35"
BLUE = "#63b5ff"
RED = "#ff7e86"


class DashboardMixin:
    """Build the single-screen OffTheBall dashboard.

    The methods mirror methods on ``gui.App`` so the mixin can be composed
    without changing the analysis worker.  All displayed numbers come from
    the current ``last_info``/``event_timeline`` state; no placeholder match
    data is generated here.
    """

    def _build(self):
        self.root.title("OffTheBall")
        self.root.geometry("1440x900")
        self.root.minsize(1100, 740)
        self.root.configure(bg=BG)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dashboard.TButton", background=SURFACE_RAISED,
                        foreground=TEXT, bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, padding=(11, 7), font=("Segoe UI", 9))
        style.configure("Dashboard.TButton", borderwidth=0, relief="flat")
        style.map("Dashboard.TButton", background=[("active", "#1c3b52"),
                                                     ("disabled", "#101c28")],
                  foreground=[("disabled", "#647786")])
        style.configure("Mint.TButton", background=MINT, foreground="#07111f",
                        bordercolor=MINT, padding=(14, 7), font=("Segoe UI", 9, "bold"))
        style.map("Mint.TButton", background=[("active", "#c5ffe0"),
                                               ("disabled", "#517862")])
        style.configure("Dashboard.Horizontal.TScale", troughcolor="#1a3246",
                        background=MINT, sliderlength=14)
        style.configure("Dashboard.Treeview", background=SURFACE, fieldbackground=SURFACE,
                        foreground=TEXT, bordercolor=BORDER, rowheight=28,
                        font=("Segoe UI", 9))
        style.configure("Dashboard.Treeview.Heading", background=SURFACE_RAISED,
                        foreground=MUTED, bordercolor=BORDER, font=("Segoe UI", 8, "bold"))
        style.configure("Dashboard.Treeview", borderwidth=0, relief="flat")
        style.layout("Dashboard.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.map("Dashboard.Treeview", background=[("selected", "#1b4050")])

        self.seek_position = tk.DoubleVar(self.root, value=0.0)
        self.time_label = tk.StringVar(self.root, value="00:00 / 00:00")
        self.metric_vars = {name: tk.StringVar(self.root, value="—") for name in
                            ("players", "team_a", "team_b", "spacing")}

        top = tk.Frame(self.root, bg=BG, height=58)
        top.pack(fill="x", padx=22, pady=(10, 5))
        top.pack_propagate(False)
        tk.Label(top, text="OffTheBall", bg=BG, fg=TEXT,
                 font=("Segoe UI", 17, "bold")).pack(side="left", padx=(0, 22))
        tk.Label(top, text="영상 분석", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 15))
        self.open_button = ttk.Button(top, text="영상 열기", style="Dashboard.TButton",
                                      command=self.open_video)
        self.open_button.pack(side="left", padx=3)
        self.run_button = ttk.Button(top, text="분석 시작", style="Mint.TButton",
                                     command=self.start_video)
        self.run_button.pack(side="left", padx=3)
        self.screen_button = ttk.Button(top, text="화면 분석", style="Dashboard.TButton",
                                        command=self.start_screen)
        self.screen_button.pack(side="left", padx=3)
        self.tablet_button = ttk.Button(top, text="태블릿", style="Dashboard.TButton",
                                        command=self.start_tablet)
        self.tablet_button.pack(side="left", padx=3)
        self.cal_button = ttk.Button(top, text="기준 보정", style="Dashboard.TButton",
                                     command=self.calibrate)
        self.cal_button.pack(side="left", padx=3)
        self.formation_button = ttk.Button(top, text="기준 배치", style="Dashboard.TButton",
                                           command=self.open_formation)
        self.formation_button.pack(side="left", padx=3)
        self.event_button = ttk.Button(top, text="이벤트 태그", style="Dashboard.TButton",
                                       command=self.tag_event)
        self.event_button.pack(side="left", padx=3)
        self.brief_button = ttk.Button(top, text="브리핑", style="Dashboard.TButton",
                                       command=self.show_briefing)
        self.brief_button.pack(side="left", padx=3)
        self.board_button = ttk.Button(top, text="전술 보드", style="Dashboard.TButton",
                                       command=self.toggle_board)
        self.help_button = ttk.Button(top, text="도움말", style="Dashboard.TButton",
                                      command=self.show_help)
        self.help_button.pack(side="right", padx=3)
        self.stop_button = ttk.Button(top, text="중지", style="Dashboard.TButton",
                                      command=self.stop.set, state="disabled")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 12))
        body.grid_columnconfigure(0, weight=2, uniform="main")
        body.grid_columnconfigure(1, weight=1, uniform="main")
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        video_card = self._card(left)
        video_card.grid(row=0, column=0, sticky="nsew")
        video_card.grid_rowconfigure(0, weight=1)
        video_card.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(video_card, bg="#020a12", width=1, height=1, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self.canvas.bind("<Configure>", lambda _event: self.paint())
        self.canvas.create_text(300, 200, text="경기 영상을 선택하세요", fill=MUTED,
                                font=("Segoe UI", 16), tags="empty")
        controls = tk.Frame(left, bg=BG)
        controls.grid(row=1, column=0, sticky="ew", pady=(9, 0))
        controls.grid_columnconfigure(1, weight=1)
        self.pause_button = ttk.Button(controls, text="▶", style="Dashboard.TButton",
                                       command=self.toggle_pause, width=8)
        self.pause_button.grid(row=0, column=0, padx=(0, 9))
        self.progress_canvas = tk.Canvas(controls, height=30, bg=SURFACE, highlightthickness=0)
        self.progress_canvas.grid(row=0, column=1, sticky="ew")
        self.progress_canvas.bind("<Button-1>", self._scrub_canvas)
        self.progress_canvas.bind("<B1-Motion>", self._scrub_canvas)
        self.progress_canvas.bind("<ButtonRelease-1>", self.seek_video)
        self.stop_button = ttk.Button(controls, text="■", width=3, style="Dashboard.TButton", command=self.stop.set, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=6)
        self.timeline_canvas = tk.Canvas(left, height=100, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        self.timeline_canvas.grid(row=3, column=0, sticky="ew", pady=(10,0))
        self.timeline_canvas.bind("<Configure>", lambda _event: self.paint_timeline())
        self.seek_scale = ttk.Scale(controls, from_=0, to=0, variable=self.seek_position,
                                    command=self._on_seek_drag,
                                    style="Dashboard.Horizontal.TScale")
        tk.Label(controls, textvariable=self.time_label, bg=BG, fg=MUTED,
                 font=("Segoe UI", 8)).grid(row=1, column=0, sticky="w")
        self.seek_scale.bind("<ButtonRelease-1>", self.seek_video)
        self.seek_scale.bind("<ButtonPress-1>", lambda e: setattr(self, "scrubbing", True))
        self.seek_scale.bind("<KeyRelease>", self.seek_video)
        tk.Label(left, textvariable=self.detail, bg=BG, fg=MUTED, anchor="w",
                 font=("Segoe UI", 9)).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)
        pitch_card = self._card(right)
        pitch_card.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        tk.Label(pitch_card, text="전술 보드  ·  105 × 68 m", bg=SURFACE,
                 fg=MUTED, anchor="w", font=("Segoe UI", 8, "bold")).pack(fill="x", padx=12, pady=(10, 5))
        self.board = tk.Canvas(pitch_card, bg="#0e5432", height=190, highlightthickness=0)
        self.board.pack(fill="x", padx=12, pady=(0, 12))
        self.board.bind("<Configure>", self._resize_pitch)

        metrics_card = self._card(right)
        metrics_card.grid(row=1, column=0, sticky="ew", pady=(0, 9))
        tk.Label(metrics_card, text="실시간 지표", bg=SURFACE, fg=MUTED,
                 anchor="w", font=("Segoe UI", 8, "bold")).pack(fill="x", padx=12, pady=(10, 6))
        metric_grid = tk.Frame(metrics_card, bg=SURFACE)
        metric_grid.pack(fill="x", padx=8, pady=(0, 9))
        labels = (("team_a", "Team A 폭"), ("team_b", "Team B 폭"),
                  ("spacing", "평균 편차"), ("players", "A 최근접 간격"))
        self._metric_grid = metric_grid
        self._metric_boxes = []
        for index, (key, label) in enumerate(labels):
            box = tk.Frame(metric_grid, bg=SURFACE_RAISED, highlightbackground=BORDER,
                           highlightthickness=1)
            box.grid(row=0, column=index, sticky="ew", padx=3, pady=3)
            metric_grid.grid_columnconfigure(index, weight=1)
            self._metric_boxes.append(box)
            tk.Label(box, text=label.upper(), bg=SURFACE_RAISED, fg=MUTED,
                     font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=8, pady=(7, 1))
            tk.Label(box, textvariable=self.metric_vars[key], bg=SURFACE_RAISED, fg=TEXT,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=8, pady=(0, 7))

        events_card = self._card(right)
        events_card.grid(row=2, column=0, sticky="nsew")
        events_card.grid_rowconfigure(1, weight=1)
        events_card.grid_columnconfigure(0, weight=1)
        tk.Label(events_card, text="이벤트 로그", bg=SURFACE, fg=MUTED,
                 anchor="w", font=("Segoe UI", 8, "bold")).grid(row=0, column=0,
                                                                  sticky="ew", padx=12, pady=(10, 6))
        log_frame = tk.Frame(events_card, bg=SURFACE)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=9, pady=(0, 9))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        self.event_tree = ttk.Treeview(log_frame, columns=("time", "event", "note"),
                                       show="headings", style="Dashboard.Treeview")
        for col, title, width in (("time", "시점", 64), ("event", "이벤트", 105), ("note", "메모", 90)):
            self.event_tree.heading(col, text=title)
            self.event_tree.column(col, width=width, anchor="w", stretch=col == "note")
        self.event_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.event_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.event_tree.configure(yscrollcommand=scrollbar.set)
        self.event_tree.bind("<Double-1>", self._seek_event)

        # Parent rendering code writes to board_log. Keep that contract while
        # making the visible event log the structured Treeview above.
        self.board_log = tk.Text(self.root, width=1, height=1, takefocus=False,
                                 state="disabled", bg=BG, fg=BG, borderwidth=0,
                                 highlightthickness=0)
        self._board_log_compat = True
        self.board_panel = pitch_card
        self._dashboard_left = left
        self._dashboard_right = right
        self._topbar = top
        self._menu_buttons = (self.screen_button, self.tablet_button, self.cal_button)
        self.root.bind("<Configure>", self._responsive_controls, add="+")
        self.render_events()
        self.paint_timeline()
        self.root.after(250, self._refresh_dashboard)
        tk.Label(self.root, textvariable=self.status, bg=BG, fg=MUTED, anchor="w", font=("Segoe UI",8)).pack(fill="x", padx=22, pady=(0,6))

    @staticmethod
    def _card(parent):
        return tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER,
                        highlightthickness=1, bd=0)

    def _resize_pitch(self, event=None):
        width = max(self.board.winfo_width(), 1)
        height = round((width-24)*68/105+24)
        if int(self.board.cget("height")) != height:
            self.board.configure(height=height)
        self.paint_board()

    def _seek_event(self, event=None):
        selected = self.event_tree.selection()
        if selected:
            index = self.event_tree.index(selected[0])
            self.seek_position.set(self.event_timeline.events[index].timestamp_s)
            self.seek_video()

    def _responsive_controls(self, event=None):
        if event is not None and event.widget is not self.root:
            return
        # Keep the primary controls usable on narrow windows; secondary input
        # actions remain available through this small overflow menu.
        if getattr(self, "_compact_menu", None) is None:
            self._compact_menu = tk.Menubutton(self._topbar, text="더보기", bg=SURFACE_RAISED,
                                               fg=TEXT, activebackground=BORDER,
                                               activeforeground=TEXT, relief="flat")
            menu = tk.Menu(self._compact_menu, tearoff=False, bg=SURFACE,
                           fg=TEXT, activebackground=BORDER)
            menu.add_command(label="화면 분석", command=self.start_screen)
            menu.add_command(label="태블릿 카메라", command=self.start_tablet)
            menu.add_command(label="기준 보정", command=self.calibrate)
            menu.add_command(label="기준 배치", command=self.open_formation)
            menu.add_command(label="경기장 선택", command=self.select_pitch)
            menu.add_command(label="브리핑", command=self.show_briefing)
            menu.add_command(label="결과 폴더", command=self.open_outputs)
            menu.add_command(label="전술 보드 표시/숨기기", command=self.toggle_board)
            menu.add_command(label="전체 영상 결과 저장", command=self.export_video)
            menu.add_separator()
            menu.add_command(label="사용 방법", command=self.show_help)
            self._compact_menu.configure(menu=menu)
        if not self._compact_menu.winfo_manager():
            self._compact_menu.pack(side="right", padx=(4, 0))
        for button in self._menu_buttons:
            button.pack_forget()
        compact_metrics = self.root.winfo_width() < 1050
        layout = 2 if compact_metrics else 4
        if getattr(self, "_metric_layout", None) != layout:
            self._metric_layout = layout
            for index, box in enumerate(self._metric_boxes):
                box.grid_configure(row=index // layout, column=index % layout)
            for column in range(4):
                self._metric_grid.grid_columnconfigure(column, weight=1 if column < layout else 0)

    def controls(self, running):
        # Event tagging, the pitch, and the briefing stay available while a
        # worker runs so the analyst can mark a live moment.
        buttons = (self.open_button, self.run_button, self.screen_button,
                   self.tablet_button)
        for button in buttons:
            button.configure(state="disabled" if running else "normal")
        # A formation is UI-only state and is safe to edit during analysis.
        # File playback can be paused and calibrated on its current raw frame.
        self.cal_button.configure(state="normal")
        self.formation_button.configure(state="normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def toggle_board(self):
        # The pitch remains in the three-card dashboard; this action simply
        # toggles the optional drawing overlay for compatibility with App.
        self._board_visible = not getattr(self, "_board_visible", True)
        if self._board_visible:
            self.board.pack(fill="x", padx=12, pady=(0, 12))
            self.board_button.configure(text="전술 보드")
            self.paint_board()
        else:
            self.board.pack_forget()
            self.board_button.configure(text="보드 표시")

    def paint_board(self):
        if not hasattr(self, "board"):
            return
        self.board.delete("all")
        width = max(self.board.winfo_width(), 260)
        height = max(self.board.winfo_height(), 140)
        # The canvas is always a 105:68 field. Letterboxing preserves scale.
        scale = min((width - 24) / 105.0, (height - 24) / 68.0)
        fw, fh = 105 * scale, 68 * scale
        left, top = (width - fw) / 2, (height - fh) / 2
        right, bottom = left + fw, top + fh
        self.board.create_rectangle(left, top, right, bottom, outline="#e6fff1", width=2)
        mid = (left + right) / 2
        self.board.create_line(mid, top, mid, bottom, fill="#e6fff1")
        self.board.create_oval(mid - 9.15*scale, (top + bottom) / 2 - 9.15*scale, mid + 9.15*scale,
                               (top + bottom) / 2 + 9.15*scale, outline="#e6fff1")
        box_w, box_h = fw * 16.5 / 105, fh * 40.3 / 68
        self.board.create_rectangle(left, (top + bottom - box_h) / 2,
                                    left + box_w, (top + bottom + box_h) / 2, outline="#e6fff1")
        self.board.create_rectangle(right - box_w, (top + bottom - box_h) / 2,
                                    right, (top + bottom + box_h) / 2, outline="#e6fff1")
        detections = getattr(self, "last_info", {}).get("detections", [])
        mapped = [item for item in detections
                  if item.get("field_xy") and len(item["field_xy"]) == 2]
        if detections and not mapped:
            calibration_status = getattr(self, "last_info", {}).get("calibration_status")
            if calibration_status == "not_calibrated" or getattr(self, "calibration", None) is None:
                message = "기준 보정 후\n선수 위치가 표시됩니다"
            else:
                message = "현재 선수 위치를\n경기장 좌표로 계산할 수 없습니다"
            self.board.create_text(width / 2, height / 2, text=message,
                                   fill="#d8eee2", justify="center",
                                   font=("Segoe UI", 10, "bold"), tags="board_status")
        for item in mapped:
            field = item["field_xy"]
            x, y = float(field[0]), float(field[1])
            px = left + max(0, min(105, x)) * scale
            py = bottom - max(0, min(68, y)) * scale
            team = item.get("team")
            color = BLUE if team == "A" else RED if team == "B" else "#c5d0d7"
            self.board.create_oval(px - 7, py - 7, px + 7, py + 7,
                                   fill=color, outline="#ffffff")
            self.board.create_text(px, py, text=str(item.get("track_id") or "?"),
                                   fill="#07111f", font=("Segoe UI", 7, "bold"))
        self._update_metrics(detections)

    def _update_metrics(self, detections):
        if not detections:
            for key in self.metric_vars:
                self.metric_vars[key].set("—")
            return
        metrics = team_metrics(detections)
        def value(team, key):
            result = metrics[team].get(key)
            return "—" if result is None else f"{result:.1f}m"
        self.metric_vars["team_a"].set(value("A", "width_m"))
        self.metric_vars["team_b"].set(value("B", "width_m"))
        deviation = formation_deviation(detections, getattr(self, "formation", None))
        self.metric_vars["spacing"].set("—" if not deviation else f"{deviation['mean_m']:.1f}m")
        self.metric_vars["players"].set(value("A", "nearest_m"))

    def render_events(self):
        if not hasattr(self, "event_tree"):
            return
        timeline = getattr(self, "event_timeline", None)
        snapshot = timeline.events if timeline is not None else ()
        if getattr(self, "_event_snapshot", None) == snapshot:
            return
        self._event_snapshot = snapshot
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        for event in timeline.events if timeline is not None else ():
            mins, secs = divmod(int(event.timestamp_s), 60)
            self.event_tree.insert("", "end", values=(f"{mins:02d}:{secs:02d}",
                                                        event.label, event.note))

    def paint_timeline(self):
        if not hasattr(self, "timeline_canvas"):
            return
        canvas = self.timeline_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 240)
        duration = max(float(getattr(self, "duration", 0.0) or 0.0), 1.0)
        canvas.create_text(14,18,text="주요 구간",fill=TEXT,anchor="w",font=("Segoe UI",10,"bold"))
        y = 52
        canvas.create_line(8, y, width - 8, y, fill=BORDER, width=4)
        position = min(1.0, max(0.0, float(self.seek_position.get()) / duration))
        x = 8 + (width - 16) * position
        canvas.create_line(8, y, x, y, fill=MINT, width=4)
        canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=MINT, outline="")
        for tick in range(7):
            tx = 8+(width-16)*tick/6
            seconds = duration*tick/6
            canvas.create_text(tx,78,text=f"{int(seconds)//60:02d}:{int(seconds)%60:02d}",fill=MUTED,font=("Segoe UI",8),anchor="w" if tick==0 else "e" if tick==6 else "center")
        progress = self.progress_canvas
        progress.delete("all")
        pw = max(progress.winfo_width(), 20)
        progress.create_line(8,15,pw-8,15,fill=BORDER,width=4)
        px = 8+(pw-16)*position
        progress.create_line(8,15,px,15,fill=MINT,width=4)
        progress.create_oval(px-5,10,px+5,20,fill=MINT,outline="")
        for event in getattr(getattr(self, "event_timeline", None), "events", ()):
            ex = 8 + (width - 16) * min(1.0, event.timestamp_s / duration)
            canvas.create_oval(ex - 3, y - 3, ex + 3, y + 3, fill=RED, outline="")

    def _on_seek_drag(self, value):
        self._set_time_label(float(value))

    def _scrub_canvas(self, event):
        self.scrubbing = True
        ratio = min(1,max(0,(event.x-8)/max(1,self.progress_canvas.winfo_width()-16)))
        self.seek_position.set(ratio*self.duration)
        self._set_time_label(self.seek_position.get())
        self.paint_timeline()

    def _set_time_label(self, seconds):
        duration = float(getattr(self, "duration", 0.0) or 0.0)
        def fmt(value):
            mins, secs = divmod(max(0, int(value)), 60)
            return f"{mins:02d}:{secs:02d}"
        self.time_label.set(f"{fmt(seconds)} / {fmt(duration)}")

    def _refresh_dashboard(self):
        """Refresh structured events and playback chrome without owning playback."""
        try:
            if not self.root.winfo_exists():
                return
            self.render_events()
            duration = float(getattr(self, "duration", 0.0) or 0.0)
            self.seek_scale.configure(to=max(0.0, duration))
            self._set_time_label(self.seek_position.get())
            self.paint_timeline()
            self.root.after(250, self._refresh_dashboard)
        except tk.TclError:
            return
