from dataclasses import dataclass

import cv2
import numpy as np

from offtheball.teams import TeamClassifier


@dataclass
class Item:
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


def frame_with_players(colors_and_boxes):
    frame = np.full((180, 280, 3), (35, 125, 45), dtype=np.uint8)
    for color, (x1, y1, x2, y2) in colors_and_boxes:
        cv2.rectangle(frame, (x1 + 5, y1 + 4), (x2 - 5, y2 - 9), color, -1)
    return frame


def scene_items():
    return [
        Item((20, 20, 70, 130), 1),
        Item((85, 20, 135, 130), 2),
        Item((150, 20, 200, 130), 3),
        Item((215, 20, 265, 130), 4),
    ]


def test_two_distinct_kits_warm_up_and_keep_scene_labels():
    red, blue = (35, 35, 205), (205, 55, 35)
    items = scene_items()
    frame = frame_with_players([(red, (20, 20, 70, 130)),
                                (blue, (85, 20, 135, 130)),
                                (red, (150, 20, 200, 130)),
                                (blue, (215, 20, 265, 130))])
    classifier = TeamClassifier(min_samples_per_team=2)

    labels = classifier.update(frame, items)
    assert classifier.ready
    assert labels == ["A", "B", "A", "B"]

    # A change in detection ordering does not swap the learned scene teams.
    next_items = [items[1], items[0], items[3], items[2]]
    next_frame = frame_with_players([(blue, (85, 20, 135, 130)),
                                     (red, (20, 20, 70, 130)),
                                     (blue, (215, 20, 265, 130)),
                                     (red, (150, 20, 200, 130))])
    assert classifier.update(next_frame, next_items) == ["B", "A", "B", "A"]


def test_track_votes_suppress_one_noisy_observation():
    red, blue = (35, 35, 205), (205, 55, 35)
    classifier = TeamClassifier(min_samples_per_team=2, vote_window=5)
    items = scene_items()
    warm = frame_with_players([(red, (20, 20, 70, 130)),
                               (blue, (85, 20, 135, 130)),
                               (red, (150, 20, 200, 130)),
                               (blue, (215, 20, 265, 130))])
    classifier.update(warm, items)

    # Track 1 is normally red.  One blue-looking frame should not relabel it.
    noisy = frame_with_players([(blue, (20, 20, 70, 130)),
                                (blue, (85, 20, 135, 130)),
                                (red, (150, 20, 200, 130)),
                                (blue, (215, 20, 265, 130))])
    assert classifier.update(noisy, items)[0] == "A"
    assert classifier.update(warm, items)[0] == "A"


def test_grass_third_kit_and_tiny_crop_are_unknown():
    red, blue = (35, 35, 205), (205, 55, 35)
    classifier = TeamClassifier(min_samples_per_team=2)
    items = scene_items()
    warm = frame_with_players([(red, (20, 20, 70, 130)),
                               (blue, (85, 20, 135, 130)),
                               (red, (150, 20, 200, 130)),
                               (blue, (215, 20, 265, 130))])
    classifier.update(warm, items)
    third = Item((20, 20, 70, 130), 99)
    tiny = Item((0, 0, 5, 8), 100)
    frame = frame_with_players([((180, 180, 40), (20, 20, 70, 130))])
    frame[:, :] = (35, 125, 45)
    assert classifier.update(frame, [third, tiny]) == [None, None]


def test_reset_tracks_keeps_centers_but_reset_forgets_scene():
    red, blue = (35, 35, 205), (205, 55, 35)
    classifier = TeamClassifier(min_samples_per_team=1)
    items = [Item((20, 20, 70, 130), 1), Item((85, 20, 135, 130), 2)]
    frame = frame_with_players([(red, (20, 20, 70, 130)),
                                (blue, (85, 20, 135, 130))])
    classifier.update(frame, items)
    assert classifier.ready
    centers = classifier.centers
    classifier.reset_tracks()
    assert classifier.ready and classifier.centers == centers
    classifier.reset()
    assert not classifier.ready
    assert classifier.centers is None

def test_blue_kit_stays_blue_under_lower_brightness():
    red, blue = (35,35,205), (205,100,65)
    items = scene_items()
    t = TeamClassifier(min_samples_per_team=2)
    warm = frame_with_players([(red,tuple(map(int,items[0].xyxy))),
                               (blue,tuple(map(int,items[1].xyxy))),
                               (red,tuple(map(int,items[2].xyxy))),
                               (blue,tuple(map(int,items[3].xyxy)))])
    expected = t.update(warm,items)[1]
    t.reset_tracks()
    dim = frame_with_players([((130,65,42),tuple(map(int,items[1].xyxy)))])
    assert t.update(dim,[items[1]]) == [expected]

