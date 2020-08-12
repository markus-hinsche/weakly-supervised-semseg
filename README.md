# Weakly Supervised Semantic Segmentation

## Setup

### Local

Create new python virtual environment. Then:

```bash
pip install -r requirements.txt
```

### Cloud (GCP)

```bash
./gcp_instance.sh
```

## Data

Find the data here: [​ISPRS Vaihingen dataset​](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)

There are 33 high-resolution (approx 1500x2000px) aerial images of the town of Vaihingen, Germany.
Labels are supplied with 1 class per pixel, as follows:

1. Impervious surfaces - WHITE
2. Building - BLUE
3. Low vegetation - TURQUOISE
4. Tree - GREEN
5. Car - YELLOW
6. Clutter/background - RED

### Data preprocessing

[src/split_to_tiles.py](src/split_to_tiles.py): Cuts 33 high-resultion images into 4497 200x200 tiles.

[src/prep_tiles.py](src/prep_tiles.py): Creates weakly-supervised tile-level annotations.

[src/split_sets.py](src/split_sets.py): Creates the N1, N2, N_validation split (randomly), see `src/constants.py` for the resulting split.

[src/mask_conversion.py](src/mask_conversion.py): Takes 3 channel RGB masks and translates them into 1 channel images where a pixel value is encoded by `0`, `1`, `2`, `3`, ..., `n_colors`

## Project

Weakly supervised learning with some fully supervised (pixel-level) annotations.

[src/fully-supervised-semseg.ipynb](src/fully-supervised-semseg.ipynb): Fully-supervised (FS) training on N1

[src/weakly-supervised-semseg.ipynb](src/weakly-supervised-semseg.ipynb): Weakly-supervised (WS) training on N2

[src/mixed-supervision-semseg.ipynb](src/mixed-supervision-semseg.ipynb): Combine FS and WS to try to improve the performance of the semantic segmentation task.

## Report and Results

See <report.md>

## Run unit tests

```bash
pytest
```
