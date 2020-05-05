# DeepKriging

## Introduction
This is a github repository for the paper entitled "DeepKriging: A Spatially Dependent Deep Neural Networks for Spatial Prediction". The paper is available in the Arxiv.

This project proposes a novel spatial prediction method by incorporating deep learning and Kriging. Rather than a simple hybrid model, we investigate the underlying relationship between the two ideas and build a direct link. The method has advantage in non-Gaussian and non-continous data over the existing Kriging-based methods in spatial statistics and allows for uncertainty quantification. The method could also contribute to the deep learning so that the spatial information, such as the user coordinates, can be properly incorporated in the DNN with a simple embedding layer. We apply the method to PM2.5 concentration.

## Implementation
The methods are implemented using "Keras" in Python. However, it can be also implemented with R interface to Keras. We also include a simple example of DeepKriging in R (DeepKriging_Example.r).

The codes for reproducibility are all in the repository. You can either run the .py file using a Python IDE or .ipynb using Jupyter Notebook. The density estimation in the DeepKriging depends on the codes in the dcdr directory, which is developed by Rui Li and Brian Reich (https://github.com/RLstat/deep-conditional-distribution-regression). The authors acknowledge their contributions in the paper.

### Author
- Yuxiao Li, Ying Sun, Brian Reich.
