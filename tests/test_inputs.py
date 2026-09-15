"""Input checks use synthetic video and mocked capture; no desktop is recorded."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from offtheball.inputs import Frame, ScreenSource, VideoSource


class InputTests(unittest.TestCase):
    def test_generated_video(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "work") as directory:
            path = Path(directory) / "clip.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened(), "MJPG encoder required for synthetic fixture")
            try:
                for value in (10, 80, 150):
                    writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            finally:
                writer.release()
            with VideoSource(path) as source:
                frames = list(source)
                self.assertAlmostEqual(source.fps, 10)
            self.assertEqual([frame.index for frame in frames], [0, 1, 2])
            self.assertEqual((frames[0].width, frames[0].height), (64, 48))
            self.assertLess(frames[0].timestamp_s, frames[1].timestamp_s)
            self.assertLessEqual(frames[0].captured_at, frames[-1].captured_at)
            self.assertLess(float(frames[0].pixels.mean()), float(frames[-1].pixels.mean()))
            with self.assertRaises(RuntimeError):
                source.read()

    def test_bad_file_and_unopened_source(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "work") as directory:
            with self.assertRaises(FileNotFoundError):
                with VideoSource(Path(directory) / "missing.avi"):
                    pass
            bad = Path(directory) / "bad.avi"
            bad.write_text("not video")
            with self.assertRaises(ValueError):
                with VideoSource(bad):
                    pass
        with self.assertRaises(RuntimeError):
            VideoSource("unopened.avi").read()

    def test_frame_rejects_invalid_dimensions(self):
        for pixels in (np.zeros((0, 2, 3), dtype=np.uint8), np.zeros((2, 2)),
                       np.zeros((2, 2, 4), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                Frame(pixels, 0, 0, 0)

    def test_region_requires_explicit_valid_dimensions(self):
        for region in (None, {}, {"left": 0, "top": 0, "width": 0, "height": 4},
                       {"left": 0.5, "top": 0, "width": 4, "height": 4}):
            with self.assertRaises(ValueError):
                ScreenSource(region)

    def test_mock_screen_pixels_are_bgr_owned_and_close(self):
        grabber = FakeGrabber()
        region = {"left": -5, "top": 0, "width": 4, "height": 3}
        with patch("offtheball.inputs._create_screen_grabber", return_value=grabber):
            with ScreenSource(region) as source:
                self.assertEqual(grabber.calls, 0)
                first, second = source.read(), source.read()
                self.assertEqual((first.width, first.height), (4, 3))
                self.assertEqual(first.pixels[0, 0].tolist(), [1, 2, 3])
                self.assertTrue(first.pixels.flags.c_contiguous)
                grabber.pixels[:] = 0
                self.assertEqual(first.pixels[0, 0].tolist(), [1, 2, 3])
                self.assertEqual(second.index, 1)
                self.assertGreaterEqual(second.timestamp_s, first.timestamp_s)
        self.assertTrue(grabber.closed)

    def test_outside_screen_bounds_closes_without_capture(self):
        grabber = FakeGrabber()
        with patch("offtheball.inputs._create_screen_grabber", return_value=grabber):
            with self.assertRaises(ValueError):
                with ScreenSource({"left": 9, "top": 0, "width": 4, "height": 3}):
                    pass
        self.assertTrue(grabber.closed)
        self.assertEqual(grabber.calls, 0)

    def test_screen_dimension_change_fails(self):
        grabber = FakeGrabber()
        grabber.pixels = np.zeros((2, 4, 4), dtype=np.uint8)
        with patch("offtheball.inputs._create_screen_grabber", return_value=grabber):
            with ScreenSource({"left": 0, "top": 0, "width": 4, "height": 3}) as source:
                with self.assertRaises(ValueError):
                    source.read()


class FakeGrabber:
    monitors = [{"left": -10, "top": 0, "width": 20, "height": 10}]

    def __init__(self):
        self.pixels = np.tile(np.array([1, 2, 3, 255], dtype=np.uint8), (3, 4, 1))
        self.calls = 0
        self.closed = False

    def grab(self, region):
        self.calls += 1
        return self.pixels

    def close(self):
        self.closed = True

