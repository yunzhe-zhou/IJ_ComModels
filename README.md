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

+ Python 3.6
    + numpy 1.18.5
    + scipy 1.5.4
    + torch 1.0.0
    + tensorflow 2.1.3
    + tensorflow-probability 0.8.0
    + sklearn 0.23.2

## File Overview
- `src/`: This folder contains all python codes used in numerical experiments and real data analysis.
  - 'inference.py' is used for data generating and supervised and generative adversarial learning.
  - `infer_utils.py` contains the utility functions to implement hypothesis testing.
  - `main.py` is an entrance to be used in command line. We can type `python main.py` to reproduce results of DRT and Sugar.
  - `main_lrt.R` is to implement the methods in ["Likelihood ratio tests for a large directed acyclic graph"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7508303/)
  - `nonlinear_learning.py` is used for structural learning of the graphs. (Refers to https://github.com/xunzheng/notears)
  - `plot.py` contains the functions to load test results and draw plots.
- `data/`: This folder where the output results and the dataset should be put.
  - 'module_name.csv' records the information of the electrode names. 

## Workflow

Follow the steps below in order to reproduce the results of this paper:
-  Put the real dataset "HCP_low.npy" and "HCP_high.npy" into the "data" folder. (Email the authors to
request the data access.) 
- `python main.py`
- `Rscript main_lrt.R`
- `python plot.py`
