# TODO Split the images & labels into tiles of size hxw=200x200

import os
from PIL import Image
from pathlib import Path

IMAGE_DATA_DIR = "data/ISPRS_semantic_labeling_Vaihingen/top/"  # file e.g. top_mosaic_09cm_area1.tif
GT_DIR = "data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/"  # file e.g. top_mosaic_09cm_area1.tif

IMAGE_DATA_TILES_DIR = "data/ISPRS_semantic_labeling_Vaihingen/top_tiles/"  # file e.g. top_mosaic_09cm_area1_tile1.tif
GT_TILES_DIR = "data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_tiles/"  # file e.g. top_mosaic_09cm_area1_tile1.tif

# https://stackoverflow.com/a/7051075/5497962
def crop(input: str, output_pattern: str, height: int, width: int):
    """Take a large image and make non-overlapping tiles (small images) out of it."""
    k=0
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j+width, i+height)
            tile = im.crop(box)
            tile.save(output_pattern % k)
            k += 1

crop(input=str(Path(IMAGE_DATA_DIR) / "top_mosaic_09cm_area1.tif"),
     output_pattern=str(Path(IMAGE_DATA_TILES_DIR) / "top_mosaic_09cm_area1_tile%s.tif"),
     height=200,
     width=200)

# TODO do for labels+ images
# TODO for all images not just one


# 33 high-res images
# ignore class 6

