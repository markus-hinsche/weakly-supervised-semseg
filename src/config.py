from typing import List

IMAGE_DATA_DIR = "data/ISPRS_semantic_labeling_Vaihingen/top/"  # file e.g. top_mosaic_09cm_area1.tif
GT_DIR = "data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/"  # file e.g. top_mosaic_09cm_area1.tif
GT_ADJ_DIR = "data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_ADJ/"  # file e.g. top_mosaic_09cm_area1.tif

IMAGE_DATA_TILES_DIR = "data/ISPRS_semantic_labeling_Vaihingen/top_tiles/"  # file e.g. top_mosaic_09cm_area1_tile1.tif
GT_TILES_DIR = "data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_tiles/"  # file e.g. top_mosaic_09cm_area1_tile1.tif
GT_ADJ_TILES_DIR = "data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_ADJ_tiles/"  # file e.g. top_mosaic_09cm_area1_tile1.tif
TILES_DIR = "data/ISPRS_semantic_labeling_Vaihingen/top_weaklabel_tiles/"  # file e.g. top_mosaic_09cm_area1_tile1_01100.tif

HEIGHT_TILE = 200
WIDTH_TILE = 200

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
TURQUOISE = (0, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)  # outlier: ignore
BLACK = (0, 0, 0)  # leftover pieces: ignore

LABELS = [
    WHITE,
    BLUE,
    TURQUOISE,
    GREEN,
    YELLOW,
]

def _gen_combinations(s_list: List[str]) -> List[str]:
    l = []
    for s in s_list: l.extend([s+'0', s+'1'])
    return l

def _gen_codes() -> List[str]:
    """returns the list ['00000', '00001' ... '11111'] """
    l = ['0', '1']
    for _ in range(4):
        l = _gen_combinations(l)
    return l
CODES = _gen_codes()


N1 = [
    'top_mosaic_09cm_area12.tif',
    'top_mosaic_09cm_area15.tif',
    'top_mosaic_09cm_area1.tif'
]

N2 = [
    'top_mosaic_09cm_area38.tif',
    'top_mosaic_09cm_area24.tif',
    'top_mosaic_09cm_area17.tif',
    'top_mosaic_09cm_area32.tif',
    'top_mosaic_09cm_area16.tif',
    'top_mosaic_09cm_area23.tif',
    'top_mosaic_09cm_area33.tif',
    'top_mosaic_09cm_area34.tif',
    'top_mosaic_09cm_area28.tif',
    'top_mosaic_09cm_area35.tif',
    'top_mosaic_09cm_area3.tif',
    'top_mosaic_09cm_area14.tif',
    'top_mosaic_09cm_area29.tif',
    'top_mosaic_09cm_area26.tif',
    'top_mosaic_09cm_area10.tif',
    'top_mosaic_09cm_area27.tif',
    'top_mosaic_09cm_area11.tif',
    'top_mosaic_09cm_area5.tif',
    'top_mosaic_09cm_area22.tif',
    'top_mosaic_09cm_area21.tif',
    'top_mosaic_09cm_area2.tif',
    'top_mosaic_09cm_area4.tif',
    'top_mosaic_09cm_area8.tif'
]

N_validation = [
    'top_mosaic_09cm_area13.tif',
    'top_mosaic_09cm_area6.tif',
    'top_mosaic_09cm_area7.tif',
    'top_mosaic_09cm_area20.tif',
    'top_mosaic_09cm_area37.tif',
    'top_mosaic_09cm_area31.tif',
    'top_mosaic_09cm_area30.tif'
]
