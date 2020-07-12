# Results

<http://www2.isprs.org/commissions/comm2/wg4/vaihingen-2d-semantic-labeling-contest.html> shows the performance of challenge competitors.

* them supervised: 91.6%
* them unsupervised: 81.8%

My results:

* me training supervised with N1+N2: 83.3% on size/2
* me training supervised with N1 only: 76.5% on size/2
* me training supervised (N1) + weakly-supervised (N2): 78% on size/2

Experiments are often done with src_size/2 which is faster (5sec instead 14sec), but yields a few percent less in accuracy.
