# DeepKriging

## Introduction
This is a github repository for reproducibility of the paper entitled "DeepKriging: A Spatially Dependent Deep Neural Networks for Spatial Prediction". The paper is available on arXiv.

This project proposes a novel spatial prediction method by combining deep learning and Kriging. Rather than a simple hybrid model, we investigate the underlying relationship between these two methods and build a direct link. The method has advantage in modeling non-Gaussian and non-continous data over the existing Kriging-based methods in spatial statistics. The method also contributes to deep learning in that the spatial dependence can be properly incorporated in the deep neural network with a simple embedding layer. We conduct two simulation studdies and apply the method to PM2.5 concentration prediction.

## Implementation
The methods are implemented using "Keras" in Python. It can be also implemented with R interface to Keras; we have included a simple example of DeepKriging in R (see 'R_example.R').

You can either run the .py file using a Python IDE or .ipynb using Jupyter Notebook. The density estimation in the DeepKriging depends on the codes in the "dcdr" directory, which is developed by Rui Li and Brian Reich (https://github.com/RLstat/deep-conditional-distribution-regression). The authors acknowledge their contribution to the paper.

### Author
- Wanfang Chen, Yuxiao Li, Brian J Reich, Ying Sun.
