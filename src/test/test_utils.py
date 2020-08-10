import glob
import math
import os, os.path
from pathlib import Path

from ..util import get_y_colors
from ..config import WHITE, BLUE, YELLOW

def test_get_y_colors():
    path = Path(__file__).parent / "test_data" /  "tile_dir" / "top_mosaic_09cm_area1_tile1_11001.tif"
    expected = [WHITE, BLUE, YELLOW]
    actual = get_y_colors(path)
    assert set(actual) == set(expected)
