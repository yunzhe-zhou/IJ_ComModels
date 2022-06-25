# The Infinitesimal Jackknife and Combinations of Models

This repository contains the implementation for the paper "The Infinitesimal Jackknife and Combinations of Models" in Python.

## Summary of the paper

The Infinitesimal Jackknife is a general method for estimating variances of parametric models, and
more recently also for some ensemble methods. In this paper we extend the Infinitesimal Jackknife to
estimate the covariance between any two models. This can be used to quantify uncertainty for combi-
nations of models, or to construct test statistics for comparing different models or ensembles of models
estimated using the same training dataset. Specific examples in this paper use boosted combinations of
models like random forests and M-estimators. We also use neural networks and XGBoost. We illustrate
the efficacy of the test through extensive simulations and real data examples (Beijing Housing), and show
theoretical consistency of the Infinitesimal Jackknife covariance estimate.

## Requirement

+ Python 3.7
    + numpy 1.21.5
    + scipy 1.7.3
    + tensorflow 2.4.0
    + keras 2.3.1
    + xgboost 1.5.1
    + tqdm 4.64.0
    + cython 0.29.30
    + matplotlib 3.5.2
    + seaborn 0.11.2

## File Overview
- `data/`: This folder contains the Beijing Housing dataset (https://www.kaggle.com/datasets/ruiqurm/lianjia) and the results of experiments will be also stored in this folder.
- `sklearn/`: This folder is used to install the revised version of sklearn by Zhengze Zhou (https://github.com/ZhengzeZhou/scikit-learn) for our experiments.

## Workflow

Follow the steps below in order to reproduce the results of this paper:
-  Put the real dataset "HCP_low.npy" and "HCP_high.npy" into the "data" folder. (Email the authors to
request the data access.) 
- `python main.py`
- `Rscript main_lrt.R`
- `python plot.py`
