import os
from pathlib import Path

from PIL import Image
import numpy as np

from .config import IMAGE_DATA_DIR, GT_DIR, IMAGE_DATA_TILES_DIR, GT_TILES_DIR, TILES_DIR, LABELS, RED, BLACK, WHITE, GT_ADJ_DIR


def convert_image(fpath, out_fpath):
    im = Image.open(fpath)
    im_array = np.asarray(im)

    print(im_array.shape)

    output_array = np.ones(im_array.shape[:2]) * 255
    print(output_array.shape)
    for i, color in enumerate(LABELS + [RED, BLACK]):
        mask = np.all(im_array == color, axis=-1)
        output_array[mask] = i

    # Make sure all pixels got converted
    assert np.all(output_array<10)

    im = Image.fromarray(output_array.astype(np.uint8), mode='L')
    im.save(out_fpath, "TIFF")

if __name__ == "__main__":

    # for fname in os.listdir(GT_DIR):
    fname = os.listdir(GT_DIR)[0]

    fpath = Path(GT_DIR) / fname
    out_fpath = Path(GT_ADJ_DIR) / fname
    convert_image(fpath, out_fpath)
