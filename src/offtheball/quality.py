"""Detect unusable black imagery without treating it as zero observed players."""
import cv2
import numpy as np


def is_black_frame(pixels):
    h, w = pixels.shape[:2]
    # Ignore browser controls, subtitles, scores and capture borders.
    y0, y1 = int(h*.1), max(int(h*.9), int(h*.1)+1)
    x0, x1 = int(w*.1), max(int(w*.9), int(w*.1)+1)
    gray = cv2.cvtColor(pixels[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return bool(np.mean(gray <= 12) >= .98 and np.mean(gray) < 8)
