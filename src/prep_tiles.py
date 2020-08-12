"""Create tile-level classification labels using the 5 classes
1. Impervious surfaces - WHITE
2. Building - BLUE
3. Low vegetation - TURQUOISE
4. Tree - GREEN
5. Car - YELLOW
6. Clutter/background - RED (will be ignored)
7. Waste - BLACK (will be ignored)

For instance a tile containing classes 1, 2, and 5, would have a label of [1, 1, 0, 0, 1].
"""

import os
from PIL import Image
from pathlib import Path
import shutil

from src.constants import IMAGE_DATA_DIR, GT_DIR, IMAGE_DATA_TILES_DIR, GT_TILES_DIR, TILES_DIR, CLASSES, ALL_CLASSES


def image_to_label(fpath_gt_tile):
    im = Image.open(fpath_gt_tile)
    distinct_pixel_values = set(im.getdata())

    assert(distinct_pixel_values.issubset(ALL_CLASSES))
    label_vector = [int(color in distinct_pixel_values) for color in CLASSES]
    return label_vector


if __name__ == "__main__":
    for fname in os.listdir(GT_TILES_DIR):
        fpath_gt_tile = Path(GT_TILES_DIR) / fname
        label_vector = image_to_label(fpath_gt_tile)
        label_string = "".join([str(l) for l in label_vector])
        print(fname, label_string)

        fpath_src = Path(IMAGE_DATA_TILES_DIR) / fname
        fpath_dest = os.path.join(TILES_DIR, f"{fname[:-4]}_{label_string}.tif")  # file e.g. top_mosaic_09cm_area1_tile167_01100.tif
        shutil.copy(fpath_src, fpath_dest)
