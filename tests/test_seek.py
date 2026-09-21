import cv2
import numpy as np
import pytest
from offtheball.inputs import VideoSource


def test_seek_forward_backward_and_end(tmp_path):
    path = tmp_path / 'seek.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), 20+i*5, np.uint8))
    writer.release()
    with VideoSource(path) as source:
        assert source.duration_s == pytest.approx(3)
        source.seek(2)
        later = source.read()
        assert later.index == 20
        assert later.timestamp_s == pytest.approx(2, abs=.1)
        source.seek(.5)
        earlier = source.read()
        assert earlier.index == 5
        assert earlier.timestamp_s == pytest.approx(.5, abs=.1)
        assert later.pixels.mean() > earlier.pixels.mean()+50
        source.seek(100)
        assert source.read().index == 29
        with pytest.raises(ValueError):
            source.seek(float('nan'))
