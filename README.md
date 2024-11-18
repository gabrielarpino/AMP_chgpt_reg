# AMP for Change Point Inference in High-Dimensional Regression
Includes code used to produce experiments in ["Inferring Change Points in High-Dimensional Regression via Approximate Message Passing"](https://arxiv.org/abs/2404.07864) by Gabriel Arpino, Xiaoqi Liu, Julia Gontarek, and Ramji Venkataramanan.

All plots are generated using Python and/or Jupyter Notebook files.

# Required Packages
In order to run the files, the following Python libraries are required: _numpy_, _jax_, _scipy_, _matplotlib_, _functools_, _tqdm_. 

# Scripts
## Minimum Working Example.ipynb
Includes a minimum working example for running AMP and inferring the locations of two change points using various different signal priors, and a uniform prior on the change point locations.  

# Linear Regression
For experiments related to linear regression, see https://github.com/gabrielarpino/AMP_chgpt_lin_reg, as well as Linear Synthetic Estimation Experiments.ipynb and Linear Synthetic Posterior Experiments.ipynb

# Logistic Regression
## Real Myocardial Infarction (MI) data example
See Myocardial Infarction Experiment.ipynb

## Synthetic Experiments
See Logistic Synthetic Estimation Experiments.ipynb and Logistic Synthetic Posterior Experiments.ipynb

# Rectified Linear Regression
See ReLU Synthetic Estimation Experiments.ipynb and ReLU Synthetic Posterior Experiments.ipynb. 
