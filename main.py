#!/usr/bin/env python
# coding: utf-8


import imageio
import torch
import numpy as np
import cv2
import os
import numpyro
import pyro.distributions as dist
import numpyro
import numpyro.distributions as n_dist
import PyQt5
import skimage.morphology as morph
import skimage

from dataProcessing import *


# In[2]:


### Load data ####
timeResolution = np.array([25, 16, 17, 15, 15, 17, 14])
spatialDownsamplingFactor=4
noVideos=2

dataPath='data/usliverseq-mp4'
videoList = loadData(dataPath,timeResolution,spatialDownsamplingFactor=spatialDownsamplingFactor,noVideos=noVideos)

### Get affine transformation ###

ptsArray = np.load('transPts.npz')["arr_0"]/spatialDownsamplingFactor

# We agree on a reference video which all other videos are transformed to, video 2, i.e. video index 1
refIdx=1

videoTransList = getAffineTrans(videoList,ptsArray,refIdx)

### Mask everything outside field of view and smooth pixels out ###

# Define smoothing parameter sigma
sigma=(0,3,3) # Do not smooth over temporal dim, only spatial
videoTransMaskList, morphsInput, vidMasks = maskAndSmooth(videoTransList,sigma)




# In[3]:


# subtract mean for all pixels
vidStd = []
for vid in videoTransMaskList:
    vidMean=vid-vid.mean(axis=0)
    vidStd.append(vidMean)



import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
import time
from numpyro.infer import MCMC, NUTS, init_to_feasible




# Carry function
def f(carry, noise_t):
  beta, z_prev, tau = carry
  z_t = jnp.matmul(beta,z_prev)+noise_t
  z_prev = z_t
  return (beta, z_prev, tau), z_t




def model(T, T_forecast, output_dim, obs=None,latent_dim=2):

    """
    Define priors over beta, tau, sigma, z_1 (keep the shapes in mind)
    """

    beta = numpyro.sample(name="beta", fn=dist.Normal(loc=jnp.zeros((latent_dim,latent_dim)), scale=jnp.ones((latent_dim,latent_dim))))
    tau = numpyro.sample(name="tau", fn=dist.HalfCauchy(scale=jnp.ones(latent_dim)))
    sigma = numpyro.sample(name="sigma", fn=dist.HalfCauchy(scale=.1))
    z_prev = numpyro.sample(name="z_prev", fn=dist.Normal(loc=jnp.zeros(latent_dim), scale=jnp.ones(latent_dim)))
    h_matrix = numpyro.sample(name="h_matrix", fn=dist.Normal(loc=jnp.ones((output_dim,latent_dim)),scale=1))
    
    """
    Define LKJ prior
    """

    L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(latent_dim, 1.))
    Sigma_lower = jnp.matmul(jnp.diag(jnp.sqrt(tau)), L_Omega) # lower cholesky factor of the covariance matrix
    noises = numpyro.sample("noises", fn=dist.MultivariateNormal(loc=jnp.zeros(latent_dim), scale_tril=Sigma_lower), sample_shape=(T+T_forecast-1,))
    
    """
    Propagate the dynamics forward using jax.lax.scan
    """

    carry = (beta, z_prev, tau)
    z_collection = [z_prev]
    carry, zs_exp = jax.lax.scan(f, carry, noises, T+T_forecast-1)
    z_collection = jnp.concatenate((jnp.array(z_collection), zs_exp), axis=0)
    
    """
    Sample the observed y (y_obs) and missing y (y_mis)
    """

    numpyro.sample(name="y_obs", fn=dist.Normal(loc=(jnp.nan_to_num(jnp.matmul(h_matrix, z_collection[:T].T)).T), scale = sigma), obs=obs)
    numpyro.sample(name="y_pred", fn=dist.Normal(loc=(jnp.nan_to_num(jnp.matmul(h_matrix, z_collection[T:].T)).T), scale = sigma), obs=None) 
 

    return z_collection


# # Test model with sin curves

data_input = vidStd[1][:500,60:80,80:100].reshape(500,-1)


# In[1]:



t0 = time.time()
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

nuts_kernel = NUTS(model=model,init_strategy=init_to_feasible())
mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=1000, num_chains=1)
mcmc.run(rng_key_, T=data_input.shape[0], T_forecast=200, output_dim=data_input.shape[1], obs=data_input,latent_dim=15)

t_fin = time.time()

print("Total time: {0:.3f}m".format((t_fin - t0)/60))


# In[ ]:


samples = {k:v for k, v in mcmc.get_samples().items()}
print(samples['y_pred'].shape)
import pickle
with open('samples.pkl', 'wb') as f:
    pickle.dump(samples, f)

