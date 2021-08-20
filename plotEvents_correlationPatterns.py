#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:05:17 2021

@author: haleigh
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from gsPython.spontaneous import corrImageFunctions

#%%
def scale_half_bool(array,scale=True):
	''' scale down boolean array by factor 2'''
	if scale:
		return np.logical_and(np.logical_and(array[...,0::2,0::2],array[...,1::2,0::2]),np.logical_and(array[...,0::2,1::2],array[...,1::2,1::2]))
	else:
		return array

def scale_half_data(array,scale=True):
	''' scale down array by factor 2'''
	if scale:
		return 0.25*array[...,0::2,0::2]+0.25*array[...,1::2,0::2]+0.25*array[...,0::2,1::2]+0.25*array[...,1::2,1::2]
	else:
		return array

def get_corr2d(corr,mask):
    npx = corr.shape[0]
    assert npx==np.sum(mask), 'ROI and correlation pattern have different number of pixels: {} vs {}'.format(np.sum(mask),npx)
    h,w = mask.shape
    corr_maps = np.zeros((npx,h,w))
    corr_maps[:,mask] = corr
    corr_maps[:,np.logical_not(mask)] = np.nan
    return corr_maps  


def getGrad(corr,roi):
    # corr = Correlation table (N x N) with N = sum(shared_roi)

    normed = (corr - np.nanmean(corr,axis=0)[None,:])/np.nanstd(corr,axis=0)[None,:]
    
    corr2d = get_corr2d(normed,roi)
    corr2d[:,~roi] = np.nan



    grad = np.nanmean(corr2d[:,delta:-delta,delta:-delta]*(corr2d[:,2*delta:,delta:-delta]+corr2d[:,delta:-delta,2*delta:]),axis=0 )/2.
    # don't bother parking gradient in a dict
    #grad_corr.update({dataset.date+'{}'.format(dataset.series_number) : grad})
    return grad

#%%
dataPath='/home/naxos2-raid1/haleigh/smithlab/Projects/Inhibition_spont/Data/'
mdlx_events=np.load(dataPath + 'mdlx_events.npy',allow_pickle=True).item()

#%%
ferret=143
events=mdlx_events[ferret]['events']  #load filtered events
roi=mdlx_events[ferret]['roi']

#%% spontaneous correlations
data_masked = scale_half_data(events,True)
roiSm = scale_half_bool(roi,True)
data_masked = data_masked[:,roiSm]

crosscorrelations = np.corrcoef(data_masked,rowvar=0)

#%% plot example events and correlation patterns
#example events
event_index=[151,120,186]
evts=events[event_index]
plt.figure()
for ievt,evt in enumerate(evts,start=1):
    plt.subplot(2,3,ievt)
    plt.imshow(evt,cmap=cm.gray)

seeds=[[48,74],[72,77],[81,103]]
for iseed,seed in enumerate(seeds,start=1):
    corrimg=corrImageFunctions.corrMatToImage(seed,crosscorrelations,roiSm)
    plt.subplot(2,3,iseed+3)
    plt.imshow(corrimg)
    plt.scatter(seed[0],seed[1],c='g')
plt.tight_layout()

plt.suptitle('Events and correlation patterns')
#%% Fractures
delta=1
grad=getGrad(crosscorrelations,roiSm)
fracture_img=1-grad

plt.figure()
plt.imshow(fracture_img,cmap=cm.gray_r,vmin=0,vmax=0.2)
plt.title('Fractures')