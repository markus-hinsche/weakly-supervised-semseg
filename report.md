# Project report

## Introduction

For the task of image segmentation, training models with pixel-level annotations (labeling each pixel in an image)
yields very good models because they can be trained with full supervision.
However, collecting images with this type of annotation is quite time consuming.

Alternatively, making high-level annotations for each image (image-level annoation)
is less labour intensive and might be able to yield comparable results.
We will refer to this as weak supervision.

Full supervision (pixel-level annotations) and weak supervision (image-level annotations) can be mixed.
In this project, we want to investigate mixed supervision and compare it to full supervision and to weak supervision.

This project explores a simple approach which designs a loss function for weak labels.

## Data and Task

We work with the [ISPRS Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html).

The dataset contains 33 high-resolution (approx 1500x2000px) aerial images of the town of Vaihingen, Germany.
Each of the aerial images are split into tiles with a resolution of 200x200.

We split the dataset into 3 sets:
* `N1` = 3 high-resolution images (10% of the data)
* `N2` = 23 high-resolution images (70% of the data)
* `N_validation` = 7 high-resolution images (20% of the data)

Now the project will train and compare semantic segmentation networks, using the following data:
* Task (i)​: N1 pixel level labels;
* Task (ii): N2 tile-level class labels
* Task (iii)​: N1 pixel level labels + N2 tile-level class labels

Tile-level classification labels are created as follows:
There are the 5 different classes (see dataset description for details)
a pixel can belong to.
Tile-level labels are encoding which classes are present in at least one pixel of the tile.

Example:
For a tile containing classes 1, 2, and 5, the label would be `[1, 1, 0, 0, 1]`.

## WeakCrossEntropy Loss function

We introduce a loss called `WeakCrossEntropy`.
We get Softmax predictions.
The loss function works as follows:
For each pixel `i`, `q_i` is the sum the predicted probabilities
of the appearing classes `y_c=1`
(we disregard classes where the label is `y_c=0`).
Now the loss for one image is defined by the formula:

<img src="https://render.githubusercontent.com/render/math?math=loss = \sum_i^N -log(q_i)">

where `N` is the number of pixels.

## Training

Task (i): We use a U-net and train it fully-supervised using N1.

Task (ii): We use a U-net and train on the N2 tile-level class labels (weakly supervised).
We don't use any pixel-level (N1) labels.
We use the `WeakCrossEntropy` loss to train it.

Task (iii): We first train the fully-supervised on N1.
We take the resulting network and continue training it weakly supervised (with tile-level class labels using `WeakCrossEntropy`)

## Results

The models were trained on a Tesla P100, with a `resnet18` backbone
and a batch size of `64`, weigth decay of `0.1`.
In fully-supervised training, the learning rate was chosen to be `3e-4`.
In purely weakly-supervised training, the learning rate was chosen to be `1e-3`.
In mixed training, the learning rate was chosen to be `3e-4` for the fully-supervised part and `1e-4` for the weakly-supervised part.

Metrics are reported in accuracy like done in the
[challenge](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html#Vaihingen2D_label_eval).
Pixels that are waste/bleed from cutting into tiles are not included when calculating accuracy.
Also, pixels annotated as background/clutter are not included.

### Best reported results from the context

For a comparison, here are the results from the best performing algorithms in the
[challenge](http://www2.isprs.org/commissions/comm2/wg4/vaihingen-2d-semantic-labeling-contest.html).
* the best supervised method achieves `91.6%` accuracy
* the best unsupervised method achieves `81.8%` accuracy

### This project's results on 100x100 resolution

Experiments are often done on a scaled-down image which is faster (approx 5sec instead 14sec), but yields a few percent less in accuracy.

* training supervised with N1+N2: `83.3%`
* Task (i): fully-supervised training with N1 only: `76.5%`
* Task (ii): weakly-supervised training with N2 only: `57.3%`
* Task (iii): mixed supervision: supervised (N1) + weakly-supervised (N2): `78%`

### This project's results on full 200x200 resolution

* training supervised with N1+N2: `85.8%`
* Task (i): fully-supervised training with N1 only: `79.6%`
* Task (ii): weakly-supervised training with N2 only: `59.3%`
* Task (iii): mixed supervision: supervised (N1) + weakly-supervised (N2): `80.5%`

This shows that for this approach on this task: mixed supervision achieves better results than weak supervision
and slightly better results than full supervision.

## Future work / Improvements

* When combining fully-supervised and weakly-supervised training,
  this project trained one epoch either fully-supervised or weakly-supervised:
  It would be interesting to use batches in a way that one batch contains both fully-supervised and weakly-supervised
  in one batch (as done in the paper [Deep Learning with Mixed Supervision for Brain Tumor Segmentation](https://arxiv.org/abs/1812.04571)).
* Another improvement could be made by having a closer look at the type of mistakes that the model makes in the predicted mask.
  Like done in the [challenge result](http://www2.isprs.org/commissions/comm2/wg4/vaihingen-2d-semantic-labeling-contest.html)
  we could examine the accuracy per class.
* This project only used Resnet18, larger U-net backbones could be tried
