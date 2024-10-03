# SAWS: Stability-based Adaptive Window Selection

This repository contains the code for implementing the algorithms and reproducing the experiments in the following paper:

Huang, Chengpiao and Wang, Kaizheng. (2023). A stability principle for learning under non-stationarity. (https://arxiv.org/abs/2310.18304).

## Algorithms

The file `algorithms.py` contain three functions that implement the algorithms in the paper:
- `SAWS_offline` implements Algorithm 1, the offline version of SAWS.
- `SAWS_online` implements Algorithm 2, the offline version of SAWS.
- `MA` implements Algorithm 3, the fixed-window benchmark.

## Problem Classes
The file `environments.py` contains the following problem classes. Problem classes designed for synthetic data contain functions that evaluate the expected loss of a decision, while those designed for real data do not. Here $\boldsymbol{z}$ or $(\boldsymbol{x},y)$ denotes the sample, $\boldsymbol{\theta}$ denotes the decision, and $\ell$ denotes the loss.

- `Gauss_env` and `real_Gauss_env` implement Gaussian mean estimation for synthetic and real data, respectively: 
$$\ell(\boldsymbol{\theta},\boldsymbol{z}) = \frac{1}{2} \\| \boldsymbol{\theta}-\boldsymbol{z} \\|_2^2.$$
- `lin_reg_env` and `real_lin_reg_env` implement linear regression for synthetic and real data, respectively:
$$\ell(\boldsymbol{\theta},(\boldsymbol{x},y)) = \frac{1}{2} (y - \boldsymbol{x}^\top\boldsymbol{\theta})^2.$$
- `logit_reg_env` implements logistic regression for synthetic data:
$$\ell(\boldsymbol{\theta},(\boldsymbol{x},y)) = \log [ 1 + \exp(\boldsymbol{x}^\top\boldsymbol{\theta}) ] - y(\boldsymbol{x}^\top\boldsymbol{\theta}).$$
- `experts_env` implements prediction with expert advice for synthetic data, where the decision set is $\\{\boldsymbol{\theta}\in\mathbb{R}_+^d:\\|\boldsymbol{\theta}\\|_1=1\\}$:
$$\ell(\boldsymbol{\theta},\boldsymbol{z}) = \boldsymbol{z}^\top \boldsymbol{\theta}.$$
- `real_newsvendor_env` implements the newsvendor problem for real data:
$$\ell(\theta,z) = h(\theta - z)\_+ + b(z - \theta)\_+.$$
- `real_quantile_env` implements quantile regression for real data:
$$\ell(\boldsymbol{\theta},(\boldsymbol{x},y)) = r(y - \boldsymbol{x}^\top\boldsymbol{\theta})\_+ + (1-r)(\boldsymbol{x}^\top\boldsymbol{\theta} - y)\_+.$$

## Experiments in the Paper

 The file `synthetic_instances.py` contains the code for implementing the two problem instances in the synthetic data experiment. The folders `synthetic`, `real_electricity_prediction`, and `real_nurse_staffing` contain the code and results for the numerical experiments in Section 7 of the paper.

 ## Citation
```
@article{HWa23,
  title={A stability principle for learning under non-stationarity},
  author={Huang, Chengpiao and Wang, Kaizheng},
  journal={arXiv preprint arXiv:2310.18304},
  year={2023}
}
```
