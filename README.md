# The Infinitesimal Jackknife and Combinations of Models

To reproduce the results, use the following command at the begining to update the branch of random forest package.
```sh
pip install git+https://github.com/ZhengzeZhou/scikit-learn.git@Subsampled-Random-Forest
```

## Dependencies
~~~~~~~~~~~~~~~~~
scikit-learn requires:
- Python (>= 2.7 or >= 3.4)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- Tensorflow (>= 2.0.0)
- xgboost (0.9.0)
~~~~~~~~~~~~~~~~~

## Instructions

Run the following code to implement model comparisons between multiple random forests, glm, neural network and XGBoost. (This is the results for Figure 1,2,3,4,5,7 of the paper.) 
~~~~~~~~~~~~~~~~~
python run_comp.py
~~~~~~~~~~~~~~~~~

Run the following code to calculate the confidence interval of neural network. (This is the results for Figure 6 of the paper.) 
~~~~~~~~~~~~~~~~~
python run_nn.py
~~~~~~~~~~~~~~~~~

To generate the figures inside this paper, use the codes in the jupyter notebook "generate_plot.ipynb".
