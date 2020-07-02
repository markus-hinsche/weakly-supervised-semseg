import glob
import math
import os, os.path
from pathlib import Path

from ..split_to_tiles import crop

def count_elements_in_dir(directory):
    return len(os.listdir(directory))

def empty_dir(directory):
    files = glob.glob(str(directory) + '/*')
    for f in files:
        os.remove(f)

def test_crop_count():
    input_dir = Path(__file__).parent / "test_data"
    output_dir = Path(__file__).parent / "test_data_tmp"

    crop(input=str(input_dir/ "minions_PNG30.PNG"),  # dim: 350 × 490
         output_pattern=str(output_dir / "minions_PNG30_tile%s.png"),
         height=100,
         width=100)
    actual = count_elements_in_dir(output_dir)
    expected = math.ceil(350 / 100) * math.ceil(490 / 100)
    assert(actual == expected)
    empty_dir(output_dir)
