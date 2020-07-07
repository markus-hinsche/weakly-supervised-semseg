import glob
import math
import os, os.path
from pathlib import Path

from ..prep_tiles import image_to_label
from ..config import LABELS, BLUE, YELLOW, WHITE


def test_prep_tile():
    fpath_gt_tile = Path(__file__).parent / "test_data" / "gt_tile.tif"
    actual = image_to_label(fpath_gt_tile)

    expected = [0, 0, 0, 0, 0]
    for color in [BLUE, YELLOW, WHITE]:
        expected[LABELS.index(color)] = 1
    assert actual == expected
