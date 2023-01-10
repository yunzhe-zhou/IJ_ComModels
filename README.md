# The Infinitesimal Jackknife and Combinations of Models

This repository contains the implementation for the paper ["The Infinitesimal Jackknife and Combinations of Models"](https://arxiv.org/abs/2209.00147) in Python.

## Summary of the paper

The Infinitesimal Jackknife is a general method for estimating variances of parametric models, and
more recently also for some ensemble methods. In this paper we extend the Infinitesimal Jackknife to
estimate the covariance between any two models. This can be used to quantify uncertainty for combi-
nations of models, or to construct test statistics for comparing different models or ensembles of models
estimated using the same training dataset. Specific examples in this paper use boosted combinations of
models like random forests and M-estimators. We also use neural networks and XGBoost. We illustrate
the efficacy of the test through extensive simulations and real data examples (Beijing Housing), and show
theoretical consistency of the Infinitesimal Jackknife covariance estimate.

## Requirements

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
- `nn_jackknife_utils.py` and 'nn_utils.py': Both files are the utils functions for calculating the IJ of neural networks.
- `utils.py`: This is the utils function for model comparisions, calculating IJ of other machine learning models and preprocessing real dataset.
- `run_nn.py`: Implements the experiments of IJ for neural networks in simulations.
- `run_real.py`: Implements the experiments for the real data analysis.
- `run_comp.py`: Implements the experiments of IJ for all the other machine learning models in simulations.
- `plot_real_result.ipynb`: Generates plots for the real data anaysis.
- `plot_sim_result.ipynb`: Generates plots for the simulations.

## Workflow

Follow the steps below in order to reproduce the results of this paper:
- Download the required packages according to the "Requirement" section.
- Use the following commands to install the revised version of sklearn package:
    + `cd sklean`
    + `chmod +x setup.py`
    + `python setup.py install`
- Use `cd ..` to navigage back to the main folder and run the following commands:
    + `python run_nn.py`
    + `python run_real.py`
    + `python run_comp.py`
- Run jupyter notebooks `plot_real_result.ipynb` and `plot_sim_result.ipynb` to reproduce the plots in this paper.

## Citation
 
Please cite our paper
["The Infinitesimal Jackknife and Combinations of Models"](https://arxiv.org/abs/2209.00147) 

``` 
@article{ghosal2022infinitesimal,
  title={The Infinitesimal Jackknife and Combinations of Models},
  author={Ghosal, Indrayudh and Zhou, Yunzhe and Hooker, Giles},
  journal={arXiv preprint arXiv:2209.00147},
  year={2022}
}
``` 
