import glob
import math
import os, os.path
from pathlib import Path

from ..util import get_y_colors, REGEX_IMG_FILE_NAME, REGEX_IMG_FILE_NAME, REGEX_IMG_FILE_NAME_WITH_LABEL_VECTOR
from ..config import WHITE, BLUE, YELLOW

def test_get_y_colors():
    path = Path(__file__).parent / "test_data" / "top_mosaic_09cm_area1_tile1_11001.tif"
    expected = [WHITE, BLUE, YELLOW]
    actual = get_y_colors(path)
    assert set(actual) == set(expected)

def test_regex_file_name():
    fname = "top_mosaic_09cm_area27_tile154.tif"
    match_result = REGEX_IMG_FILE_NAME.search(fname)
    assert match_result.group('tile_id') == "154"
    assert match_result.group('area_id') == "27"

def test_regex_file_name_with_label_vector():
    fname = "top_mosaic_09cm_area27_tile154_11001.tif"
    match_result = REGEX_IMG_FILE_NAME_WITH_LABEL_VECTOR.search(fname)
    assert match_result.group('tile_id') == "154"
    assert match_result.group('area_id') == "27"
    assert match_result.group('label_vector') == "11001"
