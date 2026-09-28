"""Programmatic hidden-window smoke; no desktop recording."""
import tkinter as tk
from unittest.mock import patch
import numpy as np
import pytest
from offtheball.gui import App, CalibrationWindow


@pytest.fixture(scope="module")
def tk_root():
    # One Tcl/Tk interpreter per process, matching the application lifecycle.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def root(tk_root):
    yield tk_root
    for job in tk_root.tk.call("after", "info"):
        tk_root.after_cancel(job)
    for widget in list(tk_root.winfo_children()):
        widget.destroy()


def test_hidden_app_and_calibration_interaction(root):
    app = App(root)
    root.update_idletasks()
    assert not app.busy()
    pixels = np.zeros((140, 240, 3), np.uint8)
    app.last_raw = pixels
    app.display = pixels
    app.paint()
    window = CalibrationWindow(root, pixels)
    window.window.withdraw()
    window.points = [[10, 10], [210, 10], [210, 110], [10, 110]]
    window.field = [[0, 0], [20, 0], [20, 10], [0, 10]]
    window.fit()
    assert window.calibration is not None
    class Click:
        x = 30
        y = 30
    window.click(Click())
    Click.x = 190
    Click.y = 90
    window.click(Click())
    assert "17.09 m" in window.note.get()
    window.reset()
    assert window.calibration is None
    window.window.destroy()


def test_open_new_file_discards_previous_pending_preview(tmp_path, root):
    import cv2
    path = tmp_path/"new.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64,48))
    assert writer.isOpened()
    writer.write(np.full((48,64,3), 90, np.uint8))
    writer.release()
    app = App(root)
    old = np.zeros((20,20,3), np.uint8)
    app.publish(old, old, {"frame_index": 500})
    with patch("offtheball.gui.filedialog.askopenfilename", return_value=str(path)):
        app.open_video()
    assert app.frames.empty()
    assert app.last_raw.shape[:2] == (48,64)


def test_input_loss_clears_cached_frame_and_canvas(root):
    app = App(root)
    pixels = np.full((40, 60, 3), 120, np.uint8)
    app.display = pixels
    app.last_raw = pixels
    app.publish(pixels, pixels, {"tablet": True, "detections": [{"x": 1}]})
    app.poll()
    assert app.canvas.find_all()
    app.events.put(("input_lost", "태블릿 카메라 스트림이 종료되었습니다."))
    app.poll()
    assert app.display is None
    assert app.last_raw is None
    assert app.canvas.find_all() == ()


def test_dashboard_layout_pitch_ratio_and_live_actions(root):
    app = App(root)
    root.deiconify()
    for size in ("1440x900", "1100x740"):
        root.geometry(size)
        root.update()
        app._resize_pitch()
        root.update_idletasks()
        assert app.canvas.winfo_width() > app._dashboard_right.winfo_width()
        assert app.event_tree.winfo_height() > 60
        assert app._compact_menu.winfo_width() > 35
        w, h = app.board.winfo_width(), app.board.winfo_height()
        assert abs((w-24)/(h-24)-105/68) < .03
        app.controls(True)
        assert str(app.event_button.cget("state")) == "normal"
        assert app.progress_canvas.winfo_width() > 100
    root.withdraw()
