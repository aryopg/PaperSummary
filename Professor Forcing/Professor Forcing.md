# Professor Forcing: A New Algorithm for Training Recurrent Networks

***Link to original paper:*** https://arxiv.org/pdf/1610.09038.pdf

## Brief Summary

Professor Forcing algorithm implements adversarial domain adaptation to encourage dynamics of the recurrent network to be the same when training the network and when sampling from the network over multiple time steps. Empirically, Professor Forcing can act as a regularizer due to the improvement of test likelihood on character level Penn Treebank and sequential MNIST. Human evaluation, and t-SNE is included.

## Introduction

### Background

- The most popular RNN training strategy is via the maximum likelihood principle, known as teacher forcing.
  - This method uses the ground truth samples $y_t$ to be fed back into the model for the prediction of later outputs. These fed back samples (hopefully) force the RNN to stay close to the ground-truth sequence.
  - It is not robust because the ground truth will not be available in the prediction phase. This procedure may result in problems in generation as small prediction error compound in the conditioning context
- To remedy the problem, scheduled sampling was introduced; mixing two kinds of inputs during training: those from the ground-truth training sequence and those generated from the model.
  - To mitigate blended generated However, when the model generates several consecutive $y_t$, it is not clear anymore that the correct target (in terms of its distribution) remains the one in the ground truth sequence.

### Contributions


## Related Works

## Model

### Experiments

## Code

```

```


[DT-RNNs]: assets/2-Figure1-1.png
[eq1]: assets/eq1.png
[eq2]: assets/eq2.png
[eq3]: assets/eq3.png
[eq3sup]: assets/eq3sup.png
[eq-spectral-radius]: assets/spectral-radius-eq.png
[eq4]: assets/eq4.png
[fig-gct]: assets/fig-gct.png
[eq5]: assets/eq5.png
[eq6]: assets/eq6.png
[eq7-9]: assets/eq7-9.png
[eq10]: assets/eq10.png
[eq11-13]: assets/eq11-13.png
[eq14]: assets/eq14.png
[eq15]: assets/eq15.png
[RHN]: assets/5-Figure3-1.png
