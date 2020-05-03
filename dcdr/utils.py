# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 23:15:02 2019

@author: Rui Li
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def _check_input(input_matrix, y_grid=None):
    if isinstance(input_matrix, pd.DataFrame):
        output_matrix = input_matrix.values
        if y_grid is None:
            y_grid = input_matrix.columns.values
    elif isinstance(input_matrix, np.ndarray):
        output_matrix = input_matrix
        if y_grid is None:
            raise ValueError('input_matrix is a numpy array, '\
                             'its corresponding grid value need to be provided')
    
    return output_matrix, y_grid


def evaluate_monotonicity(cdf, y_grid=None, return_crossing_freq=False):
    
    cdf_matrix, y_grid = _check_input(cdf, y_grid)
    nobs = cdf_matrix.shape[0]
    monotonic = []
    
    if return_crossing_freq:
        diff_matrix = np.diff(cdf_matrix)
        return np.sum(diff_matrix<0)/np.prod(diff_matrix.shape)
    else:
        for i in range(nobs):
            num_cor = pearsonr(cdf_matrix[i,:], y_grid)[0]
            denom_cor = pearsonr(np.sort(cdf_matrix[i,:]), np.sort(y_grid))[0]
            if num_cor != 0:
                mono = num_cor/denom_cor
            else:
                mono = 0
            monotonic.append(mono)
            
        return np.mean(monotonic)


def evaluate_coverage(cdf, test_y, interval, y_grid=None):
    
    cdf_matrix, y_grid = _check_input(cdf, y_grid)
    test_y = test_y.reshape(-1, 1)

    test_cvrM = cdf_to_quantile(cdf_matrix, interval, y_grid)
    cvr_indM  = np.where((test_y <= test_cvrM), 1, 0).sum(axis = 1)
    cover_percent = (cvr_indM == 1).sum()/cvr_indM.shape[0]
    
    return cover_percent


def evaluate_crps(cdf, test_y, y_grid=None):
    
    cdf_matrix, y_grid = _check_input(cdf, y_grid)
    test_y = test_y.reshape(-1, 1)
            
    ntest = test_y.shape[0]    
    test_density_gridM = np.tile(y_grid, ntest).reshape(-1, len(y_grid))
    Test_indicator_matrix = np.where((test_y <= test_density_gridM), 1, 0)
    test_score = np.mean(np.square(cdf_matrix - Test_indicator_matrix))

    return test_score


def evaluate_quantile_loss(quantile_matrix, test_y, quantiles):
    
    quantile_matrix, quantiles = _check_input(quantile_matrix, quantiles)
    test_y = test_y.flatten()
    
    if not isinstance(quantiles, list):
        if isinstance(quantiles, np.ndarray):
            quantiles = quantiles.tolist()
        else:
            quantiles = [quantiles]
    
    qt_loss = 0
    for i, qt in enumerate(quantiles):
        qt_loss += np.sum(np.where(quantile_matrix[:,i]>=test_y, 
                                   (1-qt)*np.abs(quantile_matrix[:,i]-test_y),
                                   qt*np.abs(quantile_matrix[:,i]-test_y)))
        
    test_score = qt_loss/(quantile_matrix.shape[0]*quantile_matrix.shape[1])

    return test_score


def evaluate_rmse(cdf, test_y, y_grid=None):
    
    cdf_matrix, y_grid = _check_input(cdf, y_grid)
    test_y = test_y.reshape(-1, 1)

    grid_width = np.diff(y_grid).mean()
    
    test_mean = (cdf_matrix[:,-1]*y_grid[-1] 
                - cdf_matrix[:,0]*y_grid[0] 
                - cdf_matrix.sum(axis=1)*grid_width).reshape(test_y.shape)
    
    test_score = np.sqrt(np.mean(np.square(test_y - test_mean)))

    return test_score


def cdf_to_quantile(cdf, quantiles, y_grid=None):
    
    cdf_matrix, y_grid = _check_input(cdf, y_grid)
    
    quantile_ma = np.zeros((cdf.shape[0], len(quantiles)))
    
    if y_grid.ndim > 1:
        if y_grid.shape[0] != cdf_matrix.shape[0]:
            raise ValueError('If y_grid is a two dimensional matrix, ' + 
                             'it should have same first dimension with cdf')
        for i, cdf_values in enumerate(cdf_matrix):
            quantile_ma[i,:] = np.interp(quantiles, cdf_values, y_grid[i,:])
    else:
        for i, cdf_values in enumerate(cdf_matrix):
            quantile_ma[i,:] = np.interp(quantiles, cdf_values, y_grid)        
            
    return quantile_ma 
 

def quantile_to_cdf(quantile_matrix, quantiles, y_grid):
    
    quantile_matrix, quantiles = _check_input(quantile_matrix, quantiles)
    
    cdf_ma = np.zeros((quantile_matrix.shape[0], len(y_grid)))
    
    for i, quantile_values in enumerate(quantile_matrix):
        cdf_ma[i, :] = np.interp(y_grid, quantile_values, quantiles, left=0, right=1)

    return cdf_ma
    