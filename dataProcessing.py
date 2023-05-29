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
from scipy import ndimage


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

def loadData(dataPath,timeResolution,extension='.mp4',spatialDownsamplingFactor=1,noVideos=-1):

    # Preallocate list to store videos as 3D NumPy arrays
    videoList = []

    # Get the minimum time resolution of all videos which all others should be downsampled to
    minTimeResolution = min(timeResolution)
   
    files=os.listdir(dataPath)
    # Only include files with specified extension
    files=[file for file in files if file.endswith(extension)]

    
    if noVideos!=-1: # Only include some videos
        files = files[:noVideos]
        
    for vidIdx,vidFile in enumerate(files):
        print(f'Loading video {vidIdx+1}/{len(files)}..')
        vidCv2 = cv2.VideoCapture(os.path.join(dataPath,vidFile))
        print("    Spatial resolution before spatial downsampling:" ,int(vidCv2.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(vidCv2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

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
                frameDownsampled=frame[::spatialDownsamplingFactor,::spatialDownsamplingFactor]
                frames.append(frameDownsampled)
            if ret == False:
                break
            frameIdx+=1
        print("    Spatial resolution before spatial downsampling:" ,frameDownsampled.shape[1], "x", frameDownsampled.shape[0])
        print(f'    No. of frames after temporal downsampling: {len(frames)}\n')

        video = np.stack(frames, axis=0)
        videoList.append(video)

        cv2.destroyAllWindows()
    return videoList

###########################################################
### Code used to generate transPts.npz (only used once) ###
###########################################################

# Switch to interactive mode
#%matplotlib qt 

# Identify the four corners in the ultrasound field of view of each video manually

# ptsList=[]
# for vidIdx in range(len(videosList)):
#     plt.imshow(videosList[vidIdx][0]>10,cmap='gray')
#     pts=plt.ginput(4)
#     ptsList.append(np.array(pts))
#     plt.show()
#     plt.close()
# ptsArray = np.array(ptsList).astype(np.float32)
# np.savez('transPts',ptsArray)

# Switch back to inline mode
#%matplotlib inline 
############################################################

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

def maskAndSmooth(videoList,sigmaTuple,spatialDownsamplingFactor=1):

    assert len(sigmaTuple)==3,'There must be a sigma value for all dims'

    diskSize=int(40/spatialDownsamplingFactor)

    disk = np.zeros((diskSize,diskSize))
    element = skimage.draw.disk((int(diskSize/2),int(diskSize/2)),int(9/spatialDownsamplingFactor),shape=(diskSize,diskSize))
    disk[element]=1

    morphsInput = []
    videoMasks = []
    videoTransMaskList = []
    for vid in videoList:
        vidThr = np.zeros(vid.shape[1:])
        for frame in vid:
            vidThr+=frame>10
        morphInput=vidThr>0
        vidMask = morph.opening(morphInput,disk) # First opening
        vidMask = morph.closing(vidMask,disk) # Then closing
        morphsInput.append(morphInput)
        videoMasks.append(vidMask)
        vidNormalized=vid/255
        vidSmooth=ndimage.gaussian_filter(vidNormalized,sigma=sigmaTuple)
        videoTransMaskList.append(vidSmooth*vidMask[None,:,:])

    return videoTransMaskList, morphsInput, videoMasks