#!/usr/bin/env python
# coding: utf-8

# ## First Simulation Study

# ### Load Libraries

# In[1]:


import numpy as np
import scipy.stats as stats
##Library neural nets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# Library for Gaussian process
import GPy
##Library for visualization
import matplotlib as mpl
mpl.style.use("seaborn")
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'svg'")
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import pylab 


# In[2]:


# split into input (X) and output (Y) variables
N = 1000 ##Sample Size
P = 1 ##Covariates
M = 100 ##replicates
X = np.array([np.ones(N)]).T ##Design matrix
kernel = GPy.kern.Exponential(1,1,0.1) ##Covariance Function
noise_var = 0.01 ##Nugget variance
# 1000 points evenly spaced over [0,1]
s = np.linspace(0,1,N).reshape(-1,1)
mu = np.ones(N).reshape(-1,1) # vector of the means
nugget = np.eye(N) * noise_var ##Nugget matrix
cov_mat = kernel.K(s) + nugget ##Covariance matrix
# Generate M sample path with mean mu and covariance C
np.random.seed(1)
y = np.random.multivariate_normal(mu[:,0],cov_mat,M).T


# In[3]:


print(y.shape) ##check the dimension of y


# ### Visualize the Observation

# In[4]:


plt.plot(s,y[:,0],".",mew=1.5)
plt.show()
#plt.savefig("trueGP.pdf")


# ### Create a neural network with three hidden layers

# In[5]:


def create_mlp(feature_dim):
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim = feature_dim, kernel_initializer='normal', activation='relu'))##first hidden layer
    #RBFLayer(10,initializer=InitCentersRandom(X),betas=2.0,input_shape=(num_inputs,))
    model.add(Dense(50, activation='relu'))##second hidden layer
    model.add(Dense(50, activation='relu'))##third hidden layer
    model.add(Dense(1, activation='linear'))##outpur layer
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    # Compile model
    model.compile(loss='mse', optimizer=opt, metrics=['mse','mae'])
    return model


# ### Generate basis functions

# In[6]:


num_basis = [10,19,37,73]
knots = [np.linspace(0,1,i) for i in num_basis]
##Wendland kernel
K = 0 ## basis size
phi = np.zeros((N, sum(num_basis)))
for res in range(len(num_basis)):
    theta = 1/num_basis[res]*2.5
    for i in range(num_basis[res]):
        d = np.absolute(s-knots[res][i])/theta
        for j in range(len(d)):
            if d[j] >= 0 and d[j] <= 1:
                phi[j,i + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
            else:
                phi[j,i + K] = 0
    K = K + num_basis[res]


# Check the dimension of $s$, $X$ and $y$

# In[7]:


print(s.shape)
print(X.shape)
print(y.shape)
print(phi.shape)


# ### Split the data

# In[8]:


from sklearn.model_selection import train_test_split
indices = np.arange(N)
## Split the training and testing sets
s_train, s_test, X_train, X_test, phi_train, phi_test    , y_train, y_test, idx_train, idx_test     = train_test_split(s, X, phi, y, indices, test_size=0.2)
N_train = s_train.shape[0]
N_test = s_test.shape[0]


# In[10]:


print([N_train,N_test]) 


# ** Only with X=1 **

# In[11]:


model_1 = create_mlp(feature_dim = P)
 
# train the model
print("[INFO] training model 1...")
model_1.fit(X_train, y_train[:,0], validation_split = 0.2, epochs = 100, batch_size = 32, verbose = 0)


# ** With s and X **

# In[12]:


model_2 = create_mlp(feature_dim = P + 1)
Xs_train = np.hstack((X_train,s_train)) 
# train the model
print("[INFO] training model 2...")
model_2.fit(Xs_train, y_train[:,0], validation_split = 0.2, epochs = 100, batch_size = 32, verbose = 0)


# ** With RBF and X **

# In[13]:


model_3 = create_mlp(feature_dim = P + K)
XRBF_train = np.hstack((X_train,phi_train)) 
# train the model
print("[INFO] training model 3...")
train_history = model_3.fit(XRBF_train, y_train[:,0], validation_split = 0.2, epochs = 500, batch_size = 32, verbose = 0)


# In[14]:


Xs = np.hstack((X,s))
XRBF = np.hstack((X,phi))
y0_test_1 = model_1.predict(X)
y0_test_2 = model_2.predict(Xs)
y0_test_3 = model_3.predict(XRBF)


# In[15]:


print([y0_test_1.shape,y0_test_2.shape,y0_test_3.shape])


# ### Truth from GP

# In[22]:


##Warning: it is important to write 0:1 in GPRegression to get the size (Ntrain,1)
m = GPy.models.GPRegression(s_train,y_train[:,0:1] - mu[idx_train], kernel, noise_var = noise_var)
mu_GP,var_GP = m.predict(s)
lo95_GP,up95_GP = m.predict_quantiles(s)
y0_gp = mu_GP + mu


# In[23]:


m


# In[31]:


print(y0_gp.shape)


# In[26]:


kernel2 = GPy.kern.Matern32(1,1,1)
m2 = GPy.models.GPRegression(s_train,y_train[:,0:1] - mu[idx_train],kernel2, noise_var = noise_var)
m2.optimize()
mu_GPE,var_GPE = m2.predict(s)
lo95_GPE,up95_GPE = m2.predict_quantiles(s)
y0_gpe = mu_GPE + mu


# In[27]:


m2


# In[30]:


print(y0_gpe.shape)


# ### Visualize results

# In[33]:


pylab.plot(s, y[:,0],".",label="Observation")
pylab.plot(s, y0_test_1,'blue',label="X")
pylab.plot(s, y0_test_2,'pink',label="X and s")
pylab.plot(s, y0_test_3,'red',label="X and RBFs")
pylab.plot(s, y0_gpe,'black',label="Kriging")
pylab.plot(s, y0_gp,'grey',label="Truth")
pylab.legend(loc='upper right')
pylab.show()
#plt.savefig("1D_compare.pdf")


# ### MSE, MAE, and Nonlinearity

# In[40]:


def rmse(y_true,y_pred):
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))
    return float(rmse)
def mape(y_true,y_pred):
    mape = np.mean(np.absolute(y_true-y_pred)/y_true)
    return float(mape)


# In[53]:


import time
rmse_combine = np.zeros((3,M))
mape_combine = np.zeros((3,M))
y_test0=np.zeros((2,M))
x_test0=np.zeros(M)
for i in range(M):
    print("[INFO] training %s -th replicate..." % (i+1))
    start_time1 = time.time()
    m = GPy.models.GPRegression(s_train,y_train[:,i:(i+1)] - mu[idx_train], kernel, noise_var = noise_var)
    mu_GP_test,var_GP_test = m.predict(s_test)
    y_gp_test = mu_GP_test + mu[idx_test]
    print("--- %s seconds ---" % (time.time() - start_time1))
    start_time2 = time.time()
    kernel2 = GPy.kern.Matern32(1,1,1)
    m2 = GPy.models.GPRegression(s_train,y_train[:,i:(i+1)] - mu[idx_train],kernel2, noise_var = noise_var)
    m2.optimize()
    mu_GPE_test,var_GPE_test = m2.predict(s_test)
    y_gpe_test = mu_GPE_test + mu[idx_test]
    print("--- %s seconds ---" % (time.time() - start_time2))
    start_time3 = time.time()
    model_3.fit(XRBF_train, y_train[:,i], validation_split = 0.2, epochs = 500, batch_size = 32, verbose = 0)
    y_dk_test = model_3.predict(XRBF[idx_test,:])
    print("--- %s seconds ---" % (time.time() - start_time2))
    rmse_combine[:,i] = np.array([rmse(y_test,y_gp_test),rmse(y_test,y_gpe_test),rmse(y_test,y_dk_test)])
    mape_combine[:,i] = np.array([mape(y_test,y_gp_test),mape(y_test,y_gpe_test),mape(y_test,y_dk_test)])
    y_test0[:,i:i+1]=np.array([y_dk_test[0],y_gp_test[0]])
    print(y_test0[:,i:i+1])
    print(rmse_combine[:,i])
    print(mape_combine[:,i])


# In[93]:


print(np.mean(rmse_combine,axis=1))
print(np.std(rmse_combine,axis=1))
print(np.mean(mape_combine,axis=1))
print(np.std(mape_combine,axis=1))

