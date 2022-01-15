# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 22:49:02 2018

@author: Rui Li
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .early_stopping_callback import GetBest
from .utils import (cdf_to_quantile, evaluate_monotonicity, evaluate_crps, 
evaluate_quantile_loss, evaluate_rmse, evaluate_coverage)
from keras import backend
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, SpatialDropout1D
from keras.layers import Activation, Lambda
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ReduceLROnPlateau
import gc
from scipy.stats import kstest



class Binning_CDF:
    
    def __init__(self, num_cut, hidden_list, dropout_list, seeding, 
                 cutpoint_distribution='uniform',
                 histogram_bin='fixed', loss_model='multi-binary', niter=10):
        self.num_cut = num_cut
        self.n_layer = len(hidden_list)
        self.hidden_list = hidden_list
        self.seeding = seeding
        self.histogram_bin = histogram_bin
        self.loss_model = loss_model
        self.niter = niter
        self.cutpoint_distribution = cutpoint_distribution
        if len(dropout_list) < self.n_layer:
            self.dropout_list = dropout_list + [0]*(self.n_layer - len(dropout_list))
        else:
            self.dropout_list = dropout_list[:self.n_layer]
    
    @staticmethod
    def binary_loss(y_true, y_pred):
        loss = 0
        clipped_y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)    
        loss += -tf.reduce_mean(tf.log(clipped_y_pred) * y_true)
        loss += -tf.reduce_mean(tf.log(1 - clipped_y_pred) * (1 - y_true))
        return loss
    
    @staticmethod
    def crps_loss(y_true, y_pred):
        loss = 0
        clipped_y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)    
        loss += tf.reduce_mean(tf.square(1-clipped_y_pred) * y_true)
        loss += tf.reduce_mean(tf.square(clipped_y_pred) * (1 - y_true))
        return loss
    
    @staticmethod
    def tf_cumsum(x):
        from keras import backend
        return backend.cumsum(x, axis = 1)[:,:-1]
    

    def DNNclassifier_binary(self, p, num_cut, optimizer, seeding):
        
        tf.set_random_seed(seeding)
        inputs = Input(shape=(p,))
        if isinstance(optimizer, str):
            opt = optimizer
        else:
            opt_name = optimizer.__class__.__name__
            opt_config = optimizer.get_config()
            opt_class = getattr(optimizers, opt_name)
            opt = opt_class(**opt_config)
            
        for i, n_neuron in enumerate(self.hidden_list):
            if i == 0:
                net = Dense(n_neuron, kernel_initializer = 'he_uniform')(inputs)
            else:
                net = Dense(n_neuron, kernel_initializer = 'he_uniform')(net)
            net = Activation(activation = 'elu')(net)
            net = BatchNormalization()(net)
            net = Dropout(rate=self.dropout_list[i])(net)
        
        softmaxlayer = Dense(num_cut + 1, activation='softmax', 
                       kernel_initializer = 'he_uniform')(net)
        
        output = Lambda(self.tf_cumsum)(softmaxlayer)
        model = Model(inputs = [inputs], outputs=[output])
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
        return model
    
    def DNNclassifier_crps(self, p, num_cut, optimizer, seeding):
        
        tf.set_random_seed(seeding)
        inputs = Input(shape=(p,))
        if isinstance(optimizer, str):
            opt = optimizer
        else:
            opt_name = optimizer.__class__.__name__
            opt_config = optimizer.get_config()
            opt_class = getattr(optimizers, opt_name)
            opt = opt_class(**opt_config)
        
        for i, n_neuron in enumerate(self.hidden_list):
            if i == 0:
                net = Dense(n_neuron, kernel_initializer = 'he_uniform')(inputs)
            else:
                net = Dense(n_neuron, kernel_initializer = 'he_uniform')(net)
            net = Activation(activation = 'elu')(net)
            net = BatchNormalization()(net)
            net = Dropout(rate=self.dropout_list[i])(net)
        softmaxlayer = Dense(num_cut + 1, activation='softmax', 
                       kernel_initializer = 'he_uniform')(net)
        
        output = Lambda(self.tf_cumsum)(softmaxlayer)
        model = Model(inputs = [inputs], outputs=[output])
        model.compile(optimizer=opt, loss=self.crps_loss, metrics=['accuracy'])
    
        return model
        
    def DNNclassifier_multiclass(self, p, num_cut, optimizer, seeding):
        
        tf.set_random_seed(seeding)
        inputs = Input(shape=(p,))
        if isinstance(optimizer, str):
            opt = optimizer
        else:
            opt_name = optimizer.__class__.__name__
            opt_config = optimizer.get_config()
            opt_class = getattr(optimizers, opt_name)
            opt = opt_class(**opt_config)
        
        for i, n_neuron in enumerate(self.hidden_list):
            if i == 0:
                net = Dense(n_neuron, kernel_initializer = 'he_uniform')(inputs)
            else:
                net = Dense(n_neuron, kernel_initializer = 'he_uniform')(net)
            net = Activation(activation = 'elu')(net)
            net = BatchNormalization()(net)
            net = Dropout(rate=self.dropout_list[i])(net)
            
        output = Dense(num_cut + 1, activation='softmax', 
                       kernel_initializer = 'he_uniform')(net)
        model = Model(inputs = [inputs], outputs=[output])
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
        return model
    
    @staticmethod
    def cut_generator(ncut, minimum, maximum, seed=1234, random=True, 
                      empirical_data=None, dist='uniform'):
        if random:
            np.random.seed(seed)
            if dist=='empirical' and (empirical_data is not None):
                qt_cut = np.random.uniform(0, 100, size=ncut)
                cut_points = np.percentile(empirical_data, qt_cut)
            elif dist=='uniform':
                cut_points = np.random.uniform(minimum, maximum, ncut)
        else:
            if dist=='empirical' and (empirical_data is not None):
                qt_cut = np.linspace(0, 100, num=ncut)
                cut_points = np.percentile(empirical_data, qt_cut)
            elif dist=='uniform':
                cut_points = np.linspace(minimum, maximum, num=ncut) 
                
        cut_points = np.sort(cut_points)
        
        return cut_points

    @staticmethod    
    def cut_combiner(cut_points, train_y):
        idx = np.digitize(train_y, cut_points)
        right_idx = np.unique(idx)
        left_idx = right_idx-1
        all_valid_idx = np.union1d(left_idx, right_idx)
        all_valid_idx = all_valid_idx[(all_valid_idx>=0) & (all_valid_idx<len(cut_points))]
        
        return cut_points[all_valid_idx]
        
    def fit_cdf(self, train_x, train_y, valid_x=None, valid_y=None, ylim=None, 
                batch_size = 32, epochs = 500, y_margin=0.1, opt_spec='adam',
                validation_ratio=0.2, shuffle=True, verbose=1, gpu_count=0, 
                merge_empty_bin=True):
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        
        if shuffle:
            np.random.seed(self.seeding)
            orders = np.random.permutation(train_x.shape[0])
            train_x = train_x[orders]
            train_y = train_y[orders]
            
        nobs = train_x.shape[0]
        self.p = train_x.shape[1]  

        if (valid_x is None) or (valid_y is None):
            train_len = np.ceil(nobs*(1 - validation_ratio)).astype(np.int64)
            valid_x = train_x[train_len:]
            valid_y = train_y[train_len:]
            train_x = train_x[:train_len]
            train_y = train_y[:train_len]
            
        train_y = train_y.reshape(len(train_y), -1)
        valid_y = valid_y.reshape(len(valid_y), -1)      
              
        self.x_scaler = StandardScaler()
        scaled_TrainX = self.x_scaler.fit_transform(train_x)
        scaled_ValidX = self.x_scaler.transform(valid_x)            
        
        self.y_min = np.min(train_y)
        self.y_max = np.max(train_y)
        
        if ylim is None:
            self.y_range = self.y_max - self.y_min
            self.ylim = [self.y_min - y_margin*self.y_range, self.y_max + y_margin*self.y_range]
        else:
            self.ylim = ylim.copy()

        if self.ylim[0] >= self.y_min:
            self.ylim[0] = self.y_min
        
        if self.ylim[1] <= self.y_max:
            self.ylim[1] = self.y_max

            
        np.random.seed(self.seeding)
        seedlist = np.ceil(np.random.uniform(size=self.niter)*1000000).astype(np.int64)
        
        if self.num_cut < 1:
            self.num_cut_int = np.floor(self.num_cut*nobs).astype(np.int64)
        else:
            self.num_cut_int = self.num_cut
             
        if self.histogram_bin == 'random':
            self.model_list = []
            self.random_bin_list = []
            config = tf.ConfigProto(device_count ={'GPU' : gpu_count})
            session = tf.Session(config=config)
            backend.set_session(session)             
            for i in range(self.niter):          
                seeding2 = seedlist[i]
                random_cut = self.cut_generator(self.num_cut_int, self.ylim[0], self.ylim[1], 
                                                seeding2, random=True, 
                                                empirical_data=train_y, 
                                                dist=self.cutpoint_distribution)
                
                if merge_empty_bin:
                    random_cut = self.cut_combiner(random_cut, train_y)
                    
                num_cut_actual = len(random_cut)
                random_bin  = np.insert(random_cut, 0, self.ylim[0])
                random_bin  = np.append(random_bin, self.ylim[1])
                self.random_bin_list.append(random_bin)
        
                if self.loss_model == 'multi-class':
                    Train_label = np.digitize(train_y, random_cut)
                    Valid_label = np.digitize(valid_y, random_cut)
                else: # 'multi-binary' or 'multi-crps'
                    Train_label = np.tile(random_cut, train_y.shape[0]).reshape(train_y.shape[0], -1)
                    Train_label = (Train_label > train_y).astype(np.int8)
                    Valid_label = np.tile(random_cut, valid_y.shape[0]).reshape(valid_y.shape[0], -1)
                    Valid_label = (Valid_label > valid_y).astype(np.int8)  
                    
                tf.set_random_seed(seeding2)
                
                earlyStop = GetBest(monitor='val_acc', patience = 20, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor = 0.2, patience = 7)
                callback_list = [earlyStop, reduce_lr]
                
                if self.loss_model == 'multi-class':
                    classmodel = self.DNNclassifier_multiclass(self.p, num_cut_actual, opt_spec, seeding2)
                elif self.loss_model == 'multi-binary':
                    classmodel = self.DNNclassifier_binary(self.p, num_cut_actual, opt_spec, seeding2)
                elif self.loss_model == 'multi-crps':
                    classmodel = self.DNNclassifier_crps(self.p, num_cut_actual, opt_spec, seeding2)

                classmodel.fit(scaled_TrainX, Train_label, batch_size = batch_size, 
                               epochs = epochs, callbacks = callback_list, 
                               verbose=verbose, validation_data = (scaled_ValidX, Valid_label))
                
                self.model_list.append(classmodel)

                print('The {}th iteration is run'.format(i+1))
                
        elif self.histogram_bin == 'fixed':
            self.fixed_bin_model = []
            ncut = self.num_cut_int + 2
            fixed_cut = self.cut_generator(ncut, self.ylim[0], self.ylim[1], random=False, 
                                           empirical_data=train_y, 
                                           dist=self.cutpoint_distribution)
            
            fixed_cut = fixed_cut[1:-1]
            if merge_empty_bin:
                fixed_cut = self.cut_combiner(fixed_cut, train_y)
            num_cut_actual = len(fixed_cut)
            fixed_bin  = np.insert(fixed_cut, 0, self.ylim[0])
            fixed_bin  = np.append(fixed_bin, self.ylim[1])            

            self.fixed_bin = fixed_bin
            
            if self.loss_model == 'multi-class':
                Train_label = np.digitize(train_y, fixed_cut)
                Valid_label = np.digitize(valid_y, fixed_cut)
            else: # 'multi-binary' or 'multi-crps'
                Train_label = np.tile(fixed_cut, train_y.shape[0]).reshape(train_y.shape[0], -1)
                Train_label = (Train_label > train_y).astype(np.int8)
                Valid_label = np.tile(fixed_cut, valid_y.shape[0]).reshape(valid_y.shape[0], -1)
                Valid_label = (Valid_label > valid_y).astype(np.int8)  
            
            tf.set_random_seed(self.seeding)
            
            earlyStop = GetBest(monitor='val_loss', patience = 20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 7)
            callback_list = [earlyStop, reduce_lr]
             
            if self.loss_model == 'multi-class':
                classmodel = self.DNNclassifier_multiclass(self.p, num_cut_actual, opt_spec, self.seeding)
            elif self.loss_model == 'multi-binary':
                classmodel = self.DNNclassifier_binary(self.p, num_cut_actual, opt_spec, self.seeding)
            elif self.loss_model == 'multi-crps':
                classmodel = self.DNNclassifier_crps(self.p, num_cut_actual, opt_spec, self.seeding)
            
            tf.set_random_seed(self.seeding)
            config = tf.ConfigProto(device_count = {'GPU' : gpu_count})
            session = tf.Session(config=config)
            backend.set_session(session)
            classmodel.fit(scaled_TrainX, Train_label, batch_size = batch_size, 
                           epochs = epochs, callbacks = callback_list, 
                           verbose=verbose, validation_data = (scaled_ValidX, Valid_label))
                
            self.fixed_bin_model.append(classmodel)
            

    def predict_cdf(self, test_x, y_grid=None, pred_lim=None, pred_margin=0.1, 
                    ngrid=1000, keep_cdf_matrix=True, 
                    overwrite_y_grid=True, keep_test_x=True):
        
        if y_grid is None:
            if pred_lim is None:
                if pred_margin is None:
                    pred_lim = self.ylim
                else:
                    pred_lim = [self.y_min - pred_margin*self.y_range, self.y_max + pred_margin*self.y_range]                
            
            y_grid = np.linspace(pred_lim[0], pred_lim[1], num=ngrid)
            self.pred_lim = pred_lim
        else:
            self.pred_lim = [np.min(y_grid), np.max(y_grid)]             
            
        if not isinstance(test_x, np.ndarray):
            test_x = np.array(test_x)
            
        if test_x.ndim <2:
            test_x = test_x.reshape(-1, self.p)
            
        y_grid = y_grid.flatten()
        
        scaled_test_x = self.x_scaler.transform(test_x)
        
        TestX_CDF_matrix = np.zeros((test_x.shape[0], len(y_grid)))
        
        if keep_test_x:
            self.test_x = test_x
        
        if self.histogram_bin == 'random':
            for i in range(self.niter):
                random_bin = self.random_bin_list[i]
                bin_width  = random_bin[1:] - random_bin[:-1]
                random_cut = random_bin[1:-1]
                num_cut_actual = len(random_cut)
                bin_ids    =  np.digitize(y_grid, random_cut)
                
                classmodel = self.model_list[i]
                output     = classmodel.predict(scaled_test_x)
        
                update_weight = 1/(i + 1)
            
                for j, nbin in enumerate(bin_ids):
                    
                    if y_grid[j] < self.ylim[0]:
                        cdf_v = 0
                    elif y_grid[j] > self.ylim[1]:
                        cdf_v = 1                 
                    elif self.loss_model == 'multi-binary' or self.loss_model == 'multi-crps':
                        if nbin == 0:
                            cdf_v = output[:,nbin]*(y_grid[j]-random_bin[nbin])/bin_width[nbin]
                        elif nbin < num_cut_actual:
                            cdf_v = output[:,(nbin-1)] +\
                            (output[:,nbin] - output[:,(nbin-1)]) * (
                                    y_grid[j]-random_bin[nbin])/bin_width[nbin]
                        else:
                            cdf_v = output[:, (nbin-1)] + \
                            (1 - output[:,(nbin-1)]) * (
                                    y_grid[j]-random_bin[nbin])/bin_width[nbin]                        
                    elif self.loss_model == 'multi-class':
                        if nbin == 0:
                            cdf_v = output[:,nbin]*(y_grid[j]-random_bin[nbin])/bin_width[nbin]
                        else:
                            cdf_v = output[:,:nbin].sum(axis=1) +\
                            output[:,nbin]*(y_grid[j]-random_bin[nbin])/bin_width[nbin]                       
        
                    TestX_CDF_matrix[:,j] = TestX_CDF_matrix[:,j] + \
                    (cdf_v - TestX_CDF_matrix[:,j])*update_weight
                    
        elif self.histogram_bin == 'fixed':
            bin_width  = self.fixed_bin[1:] - self.fixed_bin[:-1]
            fixed_cut = self.fixed_bin[1:-1]
            num_cut_actual = len(fixed_cut)
            bin_ids    =  np.digitize(y_grid, fixed_cut)
            
            classmodel = self.fixed_bin_model[0]
            output     = classmodel.predict(scaled_test_x)
        
            for j, nbin in enumerate(bin_ids):
                
                if y_grid[j] < self.ylim[0]:
                    cdf_v = 0
                elif y_grid[j] > self.ylim[1]:
                    cdf_v = 1                 
                elif self.loss_model == 'multi-binary' or self.loss_model == 'multi-crps':
                    if nbin == 0:
                        cdf_v = output[:,nbin]*(y_grid[j]-self.fixed_bin[nbin])/bin_width[nbin]
                    elif nbin < num_cut_actual:
                        cdf_v = output[:,(nbin-1)] +\
                        (output[:,nbin] - output[:,(nbin-1)]) * (
                                y_grid[j]-self.fixed_bin[nbin])/bin_width[nbin]
                    else:
                        cdf_v = output[:, (nbin-1)] + \
                        (1 - output[:,(nbin-1)]) * (
                                y_grid[j]-self.fixed_bin[nbin])/bin_width[nbin]                        
                elif self.loss_model == 'multi-class':
                    if nbin == 0:
                        cdf_v = output[:,nbin]*(y_grid[j]-self.fixed_bin[nbin])/bin_width[nbin]
                    else:
                        cdf_v = output[:,:nbin].sum(axis=1) +\
                        output[:,nbin]*(y_grid[j]-self.fixed_bin[nbin])/bin_width[nbin]                       
    
                TestX_CDF_matrix[:,j] = cdf_v
                
        cdf_df = pd.DataFrame(TestX_CDF_matrix, columns=y_grid)
        
        if keep_cdf_matrix:
            self.TestX_CDF_matrix = TestX_CDF_matrix
            
        if overwrite_y_grid:
            self.y_grid = y_grid
                       
        return cdf_df
    
    def clear_model_memory(self):
        if self.histogram_bin == 'random':
            del self.model_list
        else:
            del self.fixed_bin_model
            
        backend.clear_session()
        gc.collect()
        gc.collect()
            
        
    
    def predict_mean(self, test_x, y_grid=None, pred_lim=None, pred_margin=0.1, ngrid=1000):
        
        cdf_matrix = self.predict_cdf(test_x, y_grid=y_grid, ngrid=ngrid, pred_lim=pred_lim,
                                      pred_margin=pred_margin, keep_cdf_matrix=False, 
                                      overwrite_y_grid=True).values
                                      
        grid_width = np.diff(self.y_grid).mean()
        
        test_mean = (cdf_matrix[:,-1]*self.y_grid[-1] 
                    - cdf_matrix[:,0]*self.y_grid[0] 
                    - cdf_matrix.sum(axis=1)*grid_width)
        
        return test_mean     
    
    def predict_quantile(self, test_x, quantiles, y_grid=None, pred_lim=None, pred_margin=0.1, ngrid=1000):
        
        cdf_matrix = self.predict_cdf(test_x, y_grid=y_grid, ngrid=ngrid, pred_lim=pred_lim, 
                                      pred_margin=pred_margin, keep_cdf_matrix=False, 
                                      overwrite_y_grid=True).values
        
        if not isinstance(quantiles, list):
            if isinstance(quantiles, np.ndarray):
                quantiles = quantiles.tolist()
            else:
                quantiles = [quantiles]
        
        test_qtM = cdf_to_quantile(cdf_matrix, quantiles, self.y_grid)
        
        test_qt_df = pd.DataFrame(test_qtM, columns=quantiles)

        return test_qt_df 

    def plot_cdf(self, index=0, test_x=None, test_y=None, grid=None, pred_lim=None, pred_margin=0.1,
                 true_cdf_func=None, figsize=(12, 8), title=None):
        
        if test_x is None:
            cdf = self.TestX_CDF_matrix[index, :].copy()
            xval = self.test_x[index, :]
            grid = self.y_grid.copy()
        else:
            cdf = self.predict_cdf(test_x, y_grid=grid, pred_lim=pred_lim, 
                                   pred_margin=pred_margin,
                                   keep_cdf_matrix=False, 
                                   overwrite_y_grid=True,
                                   keep_test_x=False).values.flatten()
            if test_x.ndim > 1:
                xval = test_x[index,:]
            else:
                xval = test_x
            grid = self.y_grid.copy()
        
        cdf = cdf[grid.argsort()]
        grid.sort()
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(grid, cdf, label='predicted cdf', lw=3)
        
        if true_cdf_func is not None:
            true_cdf = true_cdf_func(xval, grid)
            ax.plot(grid, true_cdf, label='true cdf', lw=3)
            
        ax.legend(loc='best', prop={'size':16})
        
        if test_y is not None:
            if test_x is None:
                ax.axvline(x=test_y[index], color='black',  lw=3)
            else:
                ax.axvline(x=test_y, color='black', lw=3)

        if title:
            ax.set_title(title, fontsize=20)
            tlt = ax.title
            tlt.set_position([0.5, 1.02])
            
        ax.get_xaxis().set_tick_params(direction='out', labelsize=16)
        ax.get_yaxis().set_tick_params(direction='out', labelsize=16)
            
        ax.set_xlim(self.pred_lim)
        
        return ax

    def plot_density(self, index=0, test_x=None, test_y=None, grid=None, pred_lim=None, 
                     pred_margin=0.1, window=1, true_density_func=None, 
                     figsize=(12, 8), title=None, label=None, xlabel=None,
                     ylabel=None, figure=None):

        if test_x is None:
            cdf = self.TestX_CDF_matrix[index, :].copy()
            xval = self.test_x[index, :]
            grid = self.y_grid.copy()

        else:
            cdf = self.predict_cdf(test_x, y_grid=grid, pred_lim=pred_lim, 
                                   pred_margin=pred_margin,
                                   keep_cdf_matrix=False, 
                                   overwrite_y_grid=True,
                                   keep_test_x=False).values.flatten()
            xval = test_x
            grid = self.y_grid.copy()
            
            
        if len(grid) < 2*window + 1:
            raise ValueError('''The density of the most left {0} and the most right {1} 
                             grid points won't be plotted, so it requires at least 
                             {2} grid points to make density plot'''.format(window, window, 2*window + 1))        
        
        cdf = cdf[grid.argsort()]
        grid.sort()
        
        density_binwidth = grid[(2*window):] - grid[:-(2*window)]
        cdf_diff = cdf[(2*window):] - cdf[:-(2*window)]
        
        density = cdf_diff/density_binwidth
        
        if figure is not None:
            fig, ax = figure
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
         
        if label is None:
            label = 'predicted density'
            
        ax.plot(grid[window:-window], density, label=label, lw=3)
        
        if true_density_func is not None:
            true_density = true_density_func(xval, grid[window:-window])
            ax.plot(grid[window:-window], true_density, label='true density', lw=3)
            
        ax.legend(loc='best', prop={'size':16})
            
        if title:
            ax.set_title(title, fontsize=20)
            tlt = ax.title
            tlt.set_position([0.5, 1.02])
        
        if test_y is not None:
            if test_x is None:
                ax.axvline(x=test_y[index], color='black',  lw=3)
            else:
                ax.axvline(x=test_y, color='black', lw=3)
            
        ax.get_xaxis().set_tick_params(direction='out', labelsize=16)
        ax.get_yaxis().set_tick_params(direction='out', labelsize=16)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=18)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=18)
        
        ax.set_xlim(self.pred_lim)
        
        return (fig, ax)  
    
    def plot_PIT(self, test_x, test_y, density=True, return_cdf_value=False, block_size=None, 
                 **kwargs):
        
        if block_size is None:
    
            cdf_df = self.predict_cdf(test_x, y_grid=test_y, keep_cdf_matrix=False, 
                                      overwrite_y_grid=False)
            
            cdf_values = [cdf_df.iloc[i,i] for i in range(cdf_df.shape[0])]
        else:
            cdf_values = []
            if test_x.shape[0] % block_size == 0:
                nblocks = test_x.shape[0]//block_size
            else:
                nblocks = test_x.shape[0]//block_size + 1
                
            for b in range(nblocks):
                cdf_df = self.predict_cdf(test_x[b*block_size : (b+1)*block_size], 
                                          y_grid=test_y[b*block_size : (b+1)*block_size], 
                                          keep_cdf_matrix=False, overwrite_y_grid=False)
                
                cdf_values.extend([cdf_df.iloc[i,i] for i in range(cdf_df.shape[0])])
                
                del cdf_df
                gc.collect()
        
        fig, ax = plt.subplots(1, 1)
        ax.hist(cdf_values, density=density, **kwargs)
        if density:
            ax.axhline(y=1, color='red')        
        
        if return_cdf_value:
            return ax, cdf_values
        else:
            return ax 
    
    def ks_test(self, test_x, test_y, density=True, **kwargs):
        
        cdf_df = self.predict_cdf(test_x, y_grid=test_y, keep_cdf_matrix=False, 
                                  overwrite_y_grid=False)
        
        cdf_values = [cdf_df.iloc[i,i] for i in range(cdf_df.shape[0])]

        return kstest(cdf_values, 'uniform')
    
    def evaluate(self, test_x, test_y, y_grid=None, pred_lim=None, pred_margin=0.1, 
                 ngrid=1000, quantiles=None, interval=None, mode='CRPS'):
                                      
        if mode == 'QuantileLoss' and quantiles is not None:
            quantile_matrix = self.predict_quantile(test_x, quantiles,
                                                    y_grid=y_grid, 
                                                    pred_lim=pred_lim,
                                                    pred_margin=pred_margin,
                                                    ngrid=ngrid).values
            test_score = evaluate_quantile_loss(quantile_matrix, test_y, quantiles)
        else:
            cdf_matrix = self.predict_cdf(test_x, y_grid=y_grid, 
                                          pred_lim=pred_lim, 
                                          pred_margin=pred_margin,
                                          ngrid=ngrid).values                   
            if mode == 'CRPS':
                test_score = evaluate_crps(cdf_matrix, test_y, self.y_grid)
            elif mode == 'RMSE':            
                test_score = evaluate_rmse(cdf_matrix, test_y, self.y_grid)
            elif mode == 'Coverage' and interval is not None:
                test_score = evaluate_coverage(cdf_matrix, test_y, interval, self.y_grid)
            elif mode == 'Monotonicity':
                test_score = evaluate_monotonicity(cdf_matrix, self.y_grid)
            elif mode == 'Crossing':
                test_score = evaluate_monotonicity(cdf_matrix, self.y_grid, return_crossing_freq=True)
        
        return test_score


        



        
        
