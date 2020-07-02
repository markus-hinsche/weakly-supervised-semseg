"""Split the images & labels into tiles of size hxw=200x200"""

import os
from PIL import Image
from pathlib import Path

IMAGE_DATA_DIR = "data/ISPRS_semantic_labeling_Vaihingen/top/"  # file e.g. top_mosaic_09cm_area1.tif
GT_DIR = "data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/"  # file e.g. top_mosaic_09cm_area1.tif

IMAGE_DATA_TILES_DIR = "data/ISPRS_semantic_labeling_Vaihingen/top_tiles/"  # file e.g. top_mosaic_09cm_area1_tile1.tif
GT_TILES_DIR = "data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_tiles/"  # file e.g. top_mosaic_09cm_area1_tile1.tif

HEIGHT_TILE = 200
WIDTH_TILE = 200

# Inspired by https://stackoverflow.com/a/7051075/5497962
def crop(input: str, output_pattern: str, height: int, width: int):
    """Take a large image and make non-overlapping tiles (small images) out of it.

    Tiles from the right and lower edge can have a black leftover.
    """
    k=0
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j+width, i+height)
            tile = im.crop(box)
            tile.save(output_pattern % k)
            k += 1

if __name__ == "__main__":
    for fname in os.listdir(Path(IMAGE_DATA_DIR)):
        print(fname)

        # tile image
        crop(input=str(Path(IMAGE_DATA_DIR) / fname),
            output_pattern=str(Path(IMAGE_DATA_TILES_DIR) / fname)[:-4] + "_tile%s.tif",
            height=HEIGHT_TILE,
            width=WIDTH_TILE)

        # tile GT
        crop(input=str(Path(GT_DIR) / fname),
            output_pattern=str(Path(GT_TILES_DIR) / fname)[:-4] + "_tile%s.tif",
            height=HEIGHT_TILE,
            width=WIDTH_TILE)

    # Result: 33 high-res are split into 4497 tiles
