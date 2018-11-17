# NIPS 2018 Continual Learning Workshop Submission

Submission to the [NIPS 2018 Workshop on Continual Learning](https://sites.google.com/view/continual2018).
We will update information about the approach and the paper here in case of an acceptance at the workshop.

This repository contains code to reproduce our two main experiments:

- `digits`: Supervised domain adaptation between digit benchmark datasets. We evaluate the idea of "structural learning" for adaptation
  between small digit datasets
- `noise`: Supervised and Unsupervised adaptation between noisy datasets. We apply the same idea on the problem of adaptation to noisy
  datasets and compare our results to the performance of various domain adaptation algorithms. 
  
## Citation

In case you use code from this repository in your own work, please refer to our paper at the [NIPS 2018 Continual Learning Workshop](https://sites.google.com/view/continual2018):

```
@misc{Schneider2018,
   title={Multi-Task Generalization and Adaptation between Noisy Digit Datasets: An Empirical Study},
   author={Schneider, Steffen and Ecker, Alexander S. and Macke, Jakob H. and Bethge, Matthias},
   year={2018},
   url={https://sites.google.com/view/continual2018}
}
```

Depending on which functionality you use, you might also have a look at the [salad]() toolbox for domain adaptation and transfer learning:

```
@misc{schneider2018salad,
   title={Salad: A Toolbox for Semi-supervised Adaptive Learning Across Domains},
   author={Schneider, Steffen and Ecker, Alexander S. and Macke, Jakob H. and Bethge, Matthias},
   year={2018},
   url={https://openreview.net/forum?id=S1lTifykqm}
}
```

## Trained Models

We provide pre-trained models and training logs to check our evaluation scheme, available here:
(my.hidrive.com/share/dv2s1es8vo](https://my.hidrive.com/share/dv2s1es8vo)

## Quick Installation

You need a working [PyTorch](pytorch.org) installation.
We used version 0.4.0, but more recent versions might work as well.
Apart from that, install `salad` and clone this repository:

``
pip install torch-salad
git clone git@github.com:stes/nips2018-continual.git
``

## Experiments

### Digit Benchmarks (`train_digits`)

Train the multi-task adaptation model on the four small digit benchmarks MNIST, SVHN, SYNTH and USPS.
All images are upsampled to dimensions 32x32 and converted to 3 channel RGB images.

### Adaptation for Gaussian Noise (`train_noise_white`)

- Dataset:  `datasets_white`
- Evaluation: `eval_white`

### Adaptation for Salt and Pepper Noise (`train_noise_snp`)

- Dataset:  `datasets_snp`
- Evaluation: `eval_snp`

### Adaptation between Gaussian and Salt and Pepper Noise (`train_noise_mixed`)

- Dataset:  `datasets_mixed`
- Evaluation: `eval_mixed`

### Helper Functions (`solver`, `analysis`)

- functions for training: `solver`
- functions for evaluation and plotting: `analysis`

## References

Makes use of & extends the `salad` library for adaptive learning: [salad.domainadaptation.org](https://salad.domainadaptation.org).
We will gradually merge our experimental setups into `salad`.

## Contact

Maintained by [Steffen Schneider](http://stes.io).
