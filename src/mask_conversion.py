"""Takes 3 channel RGB masks and translates them into 1 channel images where a pixel value is encoded by `0`, `1`, `2`, `3`, ..., `n_colors`"""

import os
from pathlib import Path

from PIL import Image
import numpy as np

from src.config import IMAGE_DATA_DIR, GT_DIR, IMAGE_DATA_TILES_DIR, GT_TILES_DIR, GT_ADJ_TILES_DIR, TILES_DIR, LABELS, RED, BLACK, WHITE, GT_ADJ_DIR


def convert_image(fpath, out_fpath):
    im = Image.open(fpath)
    im_array = np.asarray(im)

    output_array = np.ones(im_array.shape[:2]) * 255

    for i, color in enumerate(LABELS + [RED, BLACK]):
        mask = np.all(im_array == color, axis=-1)
        output_array[mask] = i

    # Make sure all pixels got converted
    assert np.all(output_array<10)

    im = Image.fromarray(output_array.astype(np.uint8), mode='L')
    im.save(out_fpath, "TIFF")

if __name__ == "__main__":

    src_dir = GT_TILES_DIR
    dest_dir = GT_ADJ_TILES_DIR

    for fname in os.listdir(src_dir):
        fpath = Path(src_dir) / fname
        out_fpath = Path(dest_dir) / fname
        convert_image(fpath, out_fpath)
