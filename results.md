# Results

## Best reported results from the context

<http://www2.isprs.org/commissions/comm2/wg4/vaihingen-2d-semantic-labeling-contest.html> shows the performance of challenge competitors.

* them supervised: `91.6%`
* them unsupervised: `81.8%`

## My results on 100x100 resolution:

Experiments are often done with src_size/2 which is faster (5sec instead 14sec), but yields a few percent less in accuracy.

* training supervised with N1+N2: `83.3%`
* training supervised with N1 only: `76.5%`
* training supervised (N1) + weakly-supervised (N2): `78%`

## My results on full 200x200 resolution:

* training supervised with N1+N2: `85.8%`
* training supervised with N1 only: `79.6%`
* training supervised (N1) + weakly-supervised (N2): `80.5%`
