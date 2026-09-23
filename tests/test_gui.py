"""Programmatic hidden-window smoke; no desktop recording."""
import gc
import tkinter as tk
from unittest.mock import Mock, patch
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
    # Tk variables must be finalized on the interpreter's main thread.  The
    # expanded dialog tests otherwise leave cleanup to a later unrelated test.
    gc.collect()


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


def test_tactical_board_explains_missing_calibration(root):
    app = App(root)
    root.update_idletasks()
    app.last_info = {
        "calibration_status": "not_calibrated",
        "detections": [{"track_id": 1, "team": "A", "field_xy": None}],
    }
    app.paint_board()
    status = app.board.find_withtag("board_status")
    assert len(status) == 1
    assert "기준 보정" in app.board.itemcget(status[0], "text")

    app.calibration = object()
    app.last_info = {
        "calibration_status": "active",
        "detections": [{"track_id": 1, "team": "A", "field_xy": [52.5, 34.0]}],
    }
    app.paint_board()
    assert app.board.find_withtag("board_status") == ()


def test_formation_and_file_calibration_remain_available_during_analysis(root):
    app = App(root)
    app.controls(True)
    assert str(app.formation_button.cget("state")) == "normal"
    assert str(app.cal_button.cget("state")) == "normal"

    app.busy = lambda: True
    with patch("offtheball.gui.FormationWindow") as formation:
        app.open_formation()
    formation.assert_called_once()

    app.file_playback = True
    app.last_raw = np.full((60, 80, 3), 100, np.uint8)
    with patch("offtheball.gui.CalibrationWindow") as window:
        app.calibrate()
    window.assert_called_once()
    assert app.playback_paused.is_set()
    assert "일시정지" in app.status.get()


def test_saved_calibration_is_queued_for_active_file_session(root):
    from offtheball.calibration import ManualCalibration

    app = App(root)
    app.busy = lambda: True
    app.file_playback = True
    calibration = ManualCalibration.fit(
        [[5, 5], [75, 5], [75, 55], [5, 55]],
        [[0, 0], [105, 0], [105, 68], [0, 68]],
        image_size=(80, 60), frame_id="active-frame",
    )
    app._set_calibration(calibration)
    command, payload = app.playback_commands.get_nowait()
    assert command == "calibration"
    assert payload is app.calibration
    assert "현재 분석에 적용" in app.status.get()


def test_playback_worker_applies_calibration_without_seeking(root):
    app = App(root)
    source = Mock()
    session = Mock()
    replacement = object()
    app.playback_commands.put(("calibration", replacement))
    epoch, force_frame = app._apply_playback_commands(source, session, 4)
    assert epoch == 4
    assert not force_frame
    assert session.calibration is replacement
    source.seek.assert_not_called()


def test_help_dialog_contains_complete_workflow(root):
    app = App(root)
    app.show_help()
    dialogs = [widget for widget in root.winfo_children() if isinstance(widget, tk.Toplevel)]
    assert dialogs
    dialog = dialogs[-1]
    contents = "".join(
        widget.get("1.0", "end") for widget in dialog.winfo_children()
        if isinstance(widget, tk.Text)
    )
    assert "빠른 시작" in contents
    assert "기준 보정" in contents
    assert "기준 배치" in contents
    assert "그라운드 영역" in contents
    dialog.destroy()
