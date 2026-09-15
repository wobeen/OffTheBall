import numpy as np
from offtheball.quality import is_black_frame


def test_black_video_with_visible_controls_is_unusable():
    pixels = np.zeros((100,200,3), np.uint8)
    pixels[:8] = 255
    pixels[92:] = 255
    assert is_black_frame(pixels)


def test_green_field_is_not_black():
    pixels = np.zeros((100,200,3), np.uint8)
    pixels[:,:,1] = 100
    assert not is_black_frame(pixels)


def test_small_black_input():
    assert is_black_frame(np.zeros((1,1,3), np.uint8))
