import imageio
import torch
from IPython.display import clear_output
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PyQt5
import skimage.morphology as morph
import skimage


def downsampleTimeResolution(timeResolution,minTimeResolution,n_frames,flag):
    indices = np.round(np.linspace(0,timeResolution-1,minTimeResolution)).astype(int)
    downsample = np.zeros([timeResolution])
    downsample[indices] = 1
    if flag==1:
        downsampleFrames = np.hstack([downsample]*int(np.floor(n_frames/timeResolution))) # returns boolean array
        return downsampleFrames
    
    else:
        downsampleFrames = np.hstack([downsample]*int(np.floor(n_frames/timeResolution)))
        downsampleFramesIdx = np.where(downsampleFrames==1) # return indices of frames to sample in video
        print(downsampleFramesIdx[0].shape)
        return downsampleFramesIdx

def createData(dataPath,timeResolution,extension='.mp4'):
   
   # Preallocate list to store videos as 3D NumPy arrays
   videoList = []

   # Get the minimum time resolution of all videos which all others should be downsampled to
   minTimeResolution = min(timeResolution)
   
   for path,_,files in os.walk(dataPath):
      for vidIdx,vidFile in enumerate(files):
        print(vidFile)
        print(f'Loading video {vidIdx+1}..')
        if vidFile.endswith(extension):
            vidCv2 = cv2.VideoCapture(os.path.join(path,vidFile))
            print("Spatial resolution:" ,int(vidCv2.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(vidCv2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            n_frames = int(vidCv2.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'No. of frames before temporal downsampling: {n_frames}', '\n')
            print(f'Downsampling from {timeResolution[vidIdx]} to {np.min(timeResolution)} frames per second')

            sampleFrames = downsampleTimeResolution(timeResolution[vidIdx],minTimeResolution,n_frames,1)

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
            
            print(f'No. of frames after temporal downsampling: {len(frames)}')

            video = np.stack(frames, axis=0)
            videoList.append(video)

            cv2.destroyAllWindows()
   return videoList