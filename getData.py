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

def loadData(dataPath,timeResolution,extension='.mp4'):

    # Preallocate list to store videos as 3D NumPy arrays
    videoList = []

    # Get the minimum time resolution of all videos which all others should be downsampled to
    minTimeResolution = min(timeResolution)
   
    files=os.listdir(dataPath)
    # Only include files with specified extension
    files=[file for file in files if file.endswith(extension)]

    for vidIdx,vidFile in enumerate(files):
        print(f'Loading video {vidIdx+1}/{len(files)}..')
        vidCv2 = cv2.VideoCapture(os.path.join(dataPath,vidFile))
        print("    Spatial resolution:" ,int(vidCv2.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(vidCv2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        n_frames = int(vidCv2.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'    No. of frames before temporal downsampling: {n_frames}')
        print(f'    Downsampling from {timeResolution[vidIdx]} to {np.min(timeResolution)} frames per second')

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

        print(f'    No. of frames after temporal downsampling: {len(frames)}\n')

        video = np.stack(frames, axis=0)
        videoList.append(video)

        cv2.destroyAllWindows()
    return videoList

def getAffineTrans(videoList,ptsArray,refIdx):
    _ , refRows, refCols = videoList[refIdx].shape
    refPts = ptsArray[refIdx]

    videoTransList = []

    for vidIdx,vid in enumerate(videoList):
        if vidIdx!=refIdx:
            transPts = ptsArray[vidIdx]
            M = cv2.getPerspectiveTransform(transPts, refPts)
            
            framesTransList = []
            for frame in vid:
                frameTrans = cv2.warpPerspective(frame, M, (refCols, refRows))
                framesTransList.append(frameTrans)

            videoTransList.append(np.array(framesTransList))
        else: 
            # Append reference frame video without affine transformation
            videoTransList.append(vid)
    return videoTransList

def maskAndSmooth(videoList):

    disk = np.zeros((40,40))
    element = skimage.draw.disk((20,20),9,shape=(40,40))
    disk[element]=1

    morphsInput = []
    videoMasks = []
    videosTransMaskList = []
    for vid in videoList:
        vidThr = np.zeros(vid.shape[1:])
        for frame in vid:
            vidThr+=frame>10
        morphInput=vidThr>0
        vidMask = morph.opening(morphInput,disk) # First opening
        vidMask = morph.closing(vidMask,disk) # Then closing
        morphsInput.append(morphInput)
        videoMasks.append(vidMask)
        videosTransMaskList.append(vid*vidMask[None,:,:])
    return 0