# DeepKriging

## Introduction
This is a github repository for the paper entitled "DeepKriging: A Spatially Dependent Deep Neural Networks for Spatial Prediction". The paper is available in the Arxiv.

This project proposes a novel spatial prediction method by incorporating deep learning and Kriging. Rather than a simple hybrid model, we investigate the underlying relationship bewteen the two methods and build a direct link. On the one hand, the method could contribute to the spatial or spatio-temporal statistics since it works for a variety of data types such as non-Gaussian and non-continous data and allows for uncertainty quantification. On the other hand, it could contribute to the deep learning so that the spatial information, such as the user coordinates, can be properly incorporated in the DNN with a simple embedding layer.

The methods are implemented using "Keras" in Python. However, it can be also implemented with R interface to Keras. We also include a simple example of DeepKriging in R (DeepKriging_Example.r).

### Reference
* [1] [Stochastic Rainfall Modeling at Sub-kilometer Scale](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018WR022817)
* [2] [A frailty-contagion model for multi-site hourly precipitation driven by atmospheric covariates](https://www.sciencedirect.com/science/article/pii/S0309170815000032)

### Author
- Yuxiao Li and Ying Sun
