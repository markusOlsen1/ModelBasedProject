import imageio
import torch
import matplotlib
import matplotlib.pyplot as plt
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
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
import time
from numpyro.infer import MCMC, NUTS
import pickle 


# Preprocessing

def downsampleTimeResolution(timeResolution,minTimeResolution,n_frames,flag):
    indices = np.round(np.linspace(0,timeResolution-1,minTimeResolution)).astype(int)
    downsample = np.zeros([timeResolution])
    downsample[indices] = 1
    if flag==1:
        downsampleFrames = np.hstack([downsample]*int(np.floor(n_frames/timeResolution))) # returns boolean array
        print(downsampleFrames.shape)
        return downsampleFrames
    
    else:
        downsampleFrames = np.hstack([downsample]*int(np.floor(n_frames/timeResolution)))
        downsampleFramesIdx = np.where(downsampleFrames==1) # return indices of frames to sample in video
        print(downsampleFramesIdx[0].shape)
        return downsampleFramesIdx

def get_video_list(timeResolution,minTimeResolution,n_frames,n_videos): 
    videosList=[]
    for vidIdx in range(n_videos):
        vidCv2 = cv2.VideoCapture(f'data/usliverseq-mp4/volunteer{str(vidIdx+1).zfill(2)}.mp4')
        n_frames = int(vidCv2.get(cv2.CAP_PROP_FRAME_COUNT))
        sampleFrames = downsampleTimeResolution(timeResolution[vidIdx],minTimeResolution,n_frames,1)
        print(sampleFrames.shape)
        frameIdx = 0
        frames = []
        while(vidCv2.isOpened() and frameIdx<sampleFrames.shape[0]):
            ret, frame = vidCv2.read()
            if ret == True and sampleFrames[frameIdx]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            if ret == False:
                break
            frameIdx+=1

        video = np.stack(frames, axis=0)
        videosList.append(video)

        cv2.destroyAllWindows()
    return videosList

# Defining model: 

def f(carry, noise_t):
  beta, z_prev, tau = carry
  z_t = jnp.matmul(beta,z_prev) + noise_t
  z_prev = z_t
  return (beta, z_prev, tau), z_t


def model(T, T_forecast, output_dim, obs=None): #obs1=None, obs2=None):

    latent_dim = 2
    """
    Define priors over beta, tau, sigma, z_1 (keep the shapes in mind)
    """
    beta = numpyro.sample(name="beta", fn=dist.Normal(loc=jnp.zeros((latent_dim,latent_dim)), scale=jnp.ones((latent_dim,latent_dim))))
    tau = numpyro.sample(name="tau", fn=dist.HalfCauchy(scale=jnp.ones(latent_dim)))
    sigma = numpyro.sample(name="sigma", fn=dist.HalfCauchy(scale=.5))
    z_prev = numpyro.sample(name="z_1", fn=dist.Normal(loc=jnp.zeros(latent_dim), scale=jnp.ones(latent_dim)))

    H = numpyro.sample(name="H", fn=dist.Normal(loc=jnp.zeros((output_dim,latent_dim)),scale=jnp.ones((output_dim,latent_dim))))
    
    """
    Define LKJ prior
    """
    L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(2, 10.))
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
    numpyro.sample(name="y_pred", fn=dist.Normal(loc=(jnp.matmul(H,z_collection[T:].T)).T, scale = sigma), obs=None) 
    numpyro.sample(name="y_obs", fn=dist.Normal(loc=(jnp.matmul(H,z_collection[:T].T)).T, scale = sigma), obs=obs)
    # numpyro.sample(name="y_pred1", fn=dist.Normal(loc=z_collection[T:,0], scale = sigma), obs=None) 
    # numpyro.sample(name="y_obs1", fn=dist.Normal(loc=z_collection[:T, 0], scale = sigma), obs=obs1)
    # numpyro.sample(name="y_pred2", fn=dist.Normal(loc=z_collection[T:,1], scale = sigma), obs=None) 
    # numpyro.sample(name="y_obs2", fn=dist.Normal(loc=z_collection[:T, 1], scale = sigma), obs=obs2)
    return z_collection



# Defining variables
timeResolution = np.array([25, 16, 17, 15, 15, 17, 14])
minTimeResolution = min(timeResolution)
n_frames = np.array([14516, 4372, 4625, 4078, 3983, 4943, 4593])
n_videos = 1
frame_forecast = 20
n_warmup = 1
n_samples = 1

# Preprocessing
videoList = get_video_list(timeResolution,minTimeResolution,n_frames,n_videos)

temp = np.array(videoList[0][0:10,200:205,200:205])
temp = temp.reshape(temp.shape[0],-1)
print(temp.shape)


# Run model
t0 = time.time()

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

nuts_kernel = NUTS(model=model)
mcmc = MCMC(nuts_kernel, num_samples=n_samples, num_warmup=n_warmup, num_chains=1)

#mcmc.run(rng_key_, T=N, T_forecast=0, obs1=y_obs1, ix_mis1=ix_mis1, ix_obs1=ix_obs1, 
#         obs2=y_obs2, ix_mis2=ix_mis2, ix_obs2=ix_obs2)
mcmc.run(rng_key_, T=temp.shape[0], T_forecast=frame_forecast, output_dim=temp.shape[1], obs=temp)


t_fin = time.time()

print("Total time: {0:.3f}m".format((t_fin - t0)/60))

# Defining samples
samples = {k:v for k, v in mcmc.get_samples().items()}
print(type(mcmc))
print(type(samples))
with open("samples.pkl", "wb") as f:
   pickle.dump(samples, f)

samples['y_pred'].shape

temp = samples['y_pred']
base = np.array(videoList[0][0:20,200:205,200:205])
temp_mean = temp[:,1,:].mean(axis=0)

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.imshow(base[11,:].reshape(5,5),cmap='gray')
ax2.imshow(temp_mean.reshape(5,5),cmap='gray') 
plt.savefig("figures/samples.png")

samples.keys()


import seaborn as sns
nodes = ["H", "tau","beta"]
for node in nodes:
  if node in ("tau"):
    print(samples[node].shape,node)
    for i in range(2):
      plt.figure(figsize=(4,3))
      sns.histplot(samples[node][:,i], label=node+"%d"%i, kde=True, stat="density")
      plt.legend()
    plt.savefig(f"figures/posteriors{node}.png")
  else:
    fig, axs = plt.subplots(2,2)
    for i in range(2):
      axs[i] = sns.histplot(samples[node][:,i,i], label=node, kde=True, stat="density")
    plt.legend()
    plt.savefig(f"figures/posteriors{node}.png")


# Save the predicted frames: 
predicted_frames = temp[:,:,:].mean(axis=0)
np.save("predicted_frames.npy",predicted_frames)



