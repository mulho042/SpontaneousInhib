#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:55:48 2021

@author: Haleigh Mulholland
"""

# importing required library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
from scipy.stats import ranksums
from SpontInhib import correlationEccentricity

#get_ipython().run_line_magic('matplotlib', 'qt5')

#%% Load data
dataPath='' #Filepath to data here

mdlx_data=np.load(dataPath + 'mdlx_data.npy',allow_pickle=True).item()  #mDlx.GCaMP aniamls: Inhibitory datasets
syn_data=np.load(dataPath + 'syn_data.npy',allow_pickle=True).item()  #syn.GCaMP animals: Excitatory datasets


plotWavelengthSpectrum=False  #Will plot the centered, averaged correlation pattern of the correlation matrix. Must have 'F0143__spectrum.pkl' saved in same location as dataPath
plotEccentricityExamples=False  #Will plot example fitted ellipses, must import correlationEccentricity
#%% Initialize figure
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(10)  
#%% Actual figure starts here
ferrets=[21,60,143,150,181,182,183]
mdlx_fwhm=[]
for ferret in ferrets:
    FWTM=mdlx_data[ferret]['domain_size']
    ump=mdlx_data[ferret]['umperpixel']

    major=[]
    minor=[]
    for ix in FWTM:
        major.append(ix[:,0]*ump)
        minor.append(ix[:,1]*ump)
    
    major=np.concatenate(major,axis=0)
    minor=np.concatenate(minor,axis=0)

    mdlx_fwhm.append(minor)

    
ferrets=[12,18,28,29,77,89,101]
syn_fwhm=[]

for ferret in ferrets:
    FWTM=syn_data[ferret]['domain_size']
    ump=syn_data[ferret]['umperpixel']
    
    major=[]
    minor=[]
    for ix in FWTM:
        major.append(ix[:,0]*ump)
        minor.append(ix[:,1]*ump)
    
    major=np.concatenate(major,axis=0)
    minor=np.concatenate(minor,axis=0)
    syn_fwhm.append(minor)

mdlx_med_fwhm=[]
mdlx_err_fwhm=[]
for ar in mdlx_fwhm:
    ar=np.reshape(ar,-1)
    mdlx_med_fwhm.append(np.nanmedian(ar))
    err=np.array([[np.nanpercentile(ar,25)],[np.nanpercentile(ar,75)]])
    mdlx_err_fwhm.append(err)
syn_med_fwhm=[]
syn_err_fwhm=[]
for ar in syn_fwhm:
    syn_med_fwhm.append(np.nanmedian(ar))
    err=np.array([[np.nanpercentile(ar,25)],[np.nanpercentile(ar,75)]])
    syn_err_fwhm.append(err)


    
mdlx_med_fwhm=np.array(mdlx_med_fwhm)
syn_med_fwhm=np.array(syn_med_fwhm) 
mdlx_err_fwhm=np.squeeze(np.array(mdlx_err_fwhm))
syn_err_fwhm=np.squeeze(np.array(syn_err_fwhm))



# now plot
ax1 = plt.subplot2grid(shape=(4, 5), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid((4, 5), (1, 0))
ax3 = plt.subplot2grid(shape=(4, 5), loc=(0, 1), colspan=1,rowspan=2)



# plotting subplots
#overlapping domain sizes
ferret=181
ump=mdlx_data[ferret]['umperpixel']
FWTM=mdlx_data[ferret]['domain_size']
    
major=[]
minor=[]
for ix in FWTM:
    major.append(ix[:,0])
    minor.append(ix[:,1])

major=np.concatenate(major,axis=0)
minor=np.concatenate(minor,axis=0)

minor=np.array(minor)
major=np.array(major)

pix_mm=600/ump
a=np.ones((int(pix_mm*2),int(pix_mm*2)))*np.nan
ax1.imshow(a)

np.random.seed(5012021)
randInd=np.random.permutation(minor.shape[0])
for iEvt in range(200):
    x_width=minor[randInd[iEvt]]
    y_width=major[randInd[iEvt]]
    
    if x_width>y_width:
        major_axis=x_width
        minor_axis=y_width
    else:
        major_axis=y_width
        minor_axis=x_width

    elle = Ellipse(xy=(pix_mm,pix_mm),width=minor_axis,height=major_axis)
    ax1.add_artist(elle)
    elle.set_facecolor('none')
    elle.set_edgecolor('m')
    elle.set_linewidth(1)
    elle.set_alpha(0.1)
    
#ferret=77
ferret=77
ump=syn_data[ferret]['umperpixel']
FWTM=syn_data[ferret]['domain_size']
    
major=[]
minor=[]
for ix in FWTM:
    major.append(ix[:,0])
    minor.append(ix[:,1])

major=np.concatenate(major,axis=0)
minor=np.concatenate(minor,axis=0)

minor=np.array(minor)
major=np.array(major)

pix_mm=600/ump
a=np.ones((int(pix_mm*2),int(pix_mm*2)))*np.nan
ax2.imshow(a)

np.random.seed(5012021)
randInd=np.random.permutation(minor.shape[0])
for iEvt in range(200):
    x_width=minor[randInd[iEvt]]
    y_width=major[randInd[iEvt]]
    
    if x_width>y_width:
        major_axis=x_width
        minor_axis=y_width
    else:
        major_axis=y_width
        minor_axis=x_width

    elle = Ellipse(xy=(pix_mm,pix_mm),width=minor_axis,height=major_axis)
    ax2.add_artist(elle)
    elle.set_facecolor('none')
    elle.set_edgecolor('c')
    elle.set_linewidth(1)
    elle.set_alpha(0.1)

ax1.set_xlim(0,a.shape[0])
ax1.set_ylim(0,a.shape[0])
ax2.set_xlim(0,a.shape[0])
ax2.set_ylim(0,a.shape[0])

#group medians
mdlx_err_fwhm=np.array([[np.median(mdlx_med_fwhm)-np.percentile(mdlx_med_fwhm,25)],[np.percentile(mdlx_med_fwhm,75)-np.nanmedian(mdlx_med_fwhm)]])
syn_err_fwhm=np.array([[np.median(syn_med_fwhm)-np.percentile(syn_med_fwhm,25)],[np.percentile(syn_med_fwhm,75)-np.median(syn_med_fwhm)]])
    
print(mdlx_med_fwhm)
print(syn_med_fwhm)

ax3.scatter(np.ones(len(mdlx_med_fwhm)) ,mdlx_med_fwhm/1000,c='m',alpha=0.5)
ax3.scatter(np.ones(len(syn_med_fwhm)) * 2 ,syn_med_fwhm/1000,c='c',alpha=0.5)
ax3.scatter(1,np.median(mdlx_med_fwhm/1000),c='m',marker='s')
ax3.errorbar(1,np.median(mdlx_med_fwhm/1000),yerr=mdlx_err_fwhm/1000,c='m')
ax3.scatter(2,np.median(syn_med_fwhm/1000),c='c',marker='s')  
ax3.errorbar(2,np.median(syn_med_fwhm/1000),yerr=syn_err_fwhm/1000,c='c') 
ax3.set_xlim(0,3) 
ax3.set_ylim(0.2,0.6)
ax3.set_ylabel('FWHM (mm)')
ax3.set_xticks([1,2])
ax3.set_xticklabels(('Inhib','Excit'))

mdlx_err_fwhm=np.array([[np.percentile(mdlx_med_fwhm,25)],[np.percentile(mdlx_med_fwhm,75)]])
syn_err_fwhm=np.array([[np.percentile(syn_med_fwhm,25)],[np.percentile(syn_med_fwhm,75)]])
      
print('Mdlx fw90m median: {}'.format(np.nanmedian(mdlx_med_fwhm)))
print('mDlx IQR: {} - {}'.format(mdlx_err_fwhm[0],mdlx_err_fwhm[1]))
print('Syn fw90m median: {}'.format(np.nanmedian(syn_med_fwhm)))
print('syn IQR: {} - {}'.format(syn_err_fwhm[0],syn_err_fwhm[1]))
stat,pval=ranksums(mdlx_med_fwhm,syn_med_fwhm)
print('Rank sum fw90m: {:.3f}'.format(pval))
#%%
#% Dimensionality
mdlx_dim=[]
mdlx_var=[]
ferrets=[21,60,143,150,181,182,183]
for ferret in ferrets:
    dimensionality=mdlx_data[ferret]['dimensionality']
    dimensionality=np.nanmean(dimensionality)
    mdlx_dim.append(dimensionality)
    var=mdlx_data[ferret]['variance_explained']
    var=np.nanmean(var,axis=0)
    mdlx_var.append(var)
    
    
syn_dim=[]
syn_var=[]
ferrets=[12,18,28,29,77,89,101]
for ferret in ferrets:
    dimensionality=syn_data[ferret]['dimensionality']
    dimensionality=np.nanmean(dimensionality)
    syn_dim.append(dimensionality)
    var=syn_data[ferret]['variance_explained']
    var=np.nanmean(var,axis=0)
    syn_var.append(var)

mdlx_dim=np.array(mdlx_dim)
syn_dim=np.array(syn_dim)

mdlx_err=np.array([[np.median(mdlx_dim)-np.percentile(mdlx_dim,25)],[np.percentile(mdlx_dim,75)-np.median(mdlx_dim)]])
syn_err=np.array([[np.median(syn_dim)-np.percentile(syn_dim,25)],[np.percentile(syn_dim,75)-np.median(syn_dim)]])


# Now plot
fig.set_figheight(8)
fig.set_figwidth(10)  #4x6 for a 2x2 group subplot size
  

ax1 = plt.subplot2grid(shape=(4, 5), loc=(0, 2), colspan=1)
ax2 = plt.subplot2grid((4, 5), (1, 2))
ax3 = plt.subplot2grid(shape=(4, 5), loc=(0, 3), colspan=1,rowspan=2)

# plotting subplots
# mdlx individual dimensionality variance explained
for var in mdlx_var:
    ax1.plot(np.cumsum(var),c='m',alpha=0.5)
ax1.set_title('Inhib')
ax1.set_xlim(0,30)

#syn individual dimensionality variance explained
for var in syn_var:
    ax2.plot(np.cumsum(var),c='c',alpha=0.5)
ax2.set_title('Excit')
ax2.set_xlim(0,30)
ax2.set_xlabel('Component')

#Group dimensionality
ax3.scatter(np.ones(len(mdlx_dim)) ,mdlx_dim,c='m',alpha=0.5)
ax3.scatter(np.ones(len(syn_dim)) * 2 ,syn_dim,c='c',alpha=0.5)
ax3.scatter(1,np.nanmedian(mdlx_dim),c='m',marker='s')
ax3.errorbar(1,np.nanmedian(mdlx_dim),yerr=mdlx_err,c='m')
ax3.scatter(2,np.nanmedian(syn_dim),c='c',marker='s')  
ax3.errorbar(2,np.nanmedian(syn_dim),yerr=syn_err,c='c')
ax3.set_xlim(0,3)
ax3.set_ylim(5,15)
ax3.set_ylabel('Dimensionality')
ax3.set_xticks([1,2])
ax3.set_xticklabels(('Inhib','Excit'))

print('Mdlx dimensionality median: {}'.format(np.nanmedian(mdlx_dim)))
print('Mdlx IQR: {} - {}'.format(np.percentile(mdlx_dim,25),np.percentile(mdlx_dim,75)))
print('Syn dimensionality median: {}'.format(np.nanmedian(syn_dim)))
print('Syn IQR: {} - {}'.format(np.percentile(syn_dim,25),np.percentile(syn_dim,75)))
stat,pval=ranksums(mdlx_dim,syn_dim)
print('Rank sum Dimensionality: {:.3f}'.format(pval))
#%% Correlation strength
mdlx_corr=[]
ferrets=[21,60,143,150,181,182,183]
for ferret in ferrets:
    corr_strength=mdlx_data[ferret]['correlation_strength']
    mdlx_corr.append(corr_strength)

syn_corr=[]
ferrets=[12,18,28,29,77,89,101]
for ferret in ferrets:
    corr_strength=syn_data[ferret]['correlation_strength']
    syn_corr.append(corr_strength)
    
mdlx_corr=np.array(mdlx_corr)
syn_corr=np.array(syn_corr)
mdlx_err=np.array([[np.median(mdlx_corr)-np.percentile(mdlx_corr,25)],[np.percentile(mdlx_corr,75)-np.median(mdlx_corr)]])
syn_err=np.array([[np.median(syn_corr)-np.percentile(syn_corr,25)],[np.percentile(syn_corr,75)-np.median(syn_corr)]])

#plot
fig.set_figheight(8)
fig.set_figwidth(10)  #4x6 for a 2x2 group subplot size
  
ax3 = plt.subplot2grid(shape=(4, 5), loc=(0, 4), colspan=1,rowspan=2)


ax3.scatter(np.ones(len(mdlx_corr)) ,mdlx_corr,c='m',alpha=0.5)
ax3.scatter(np.ones(len(syn_corr)) * 2 ,syn_corr,c='c',alpha=0.5)
ax3.scatter(1,np.median(mdlx_corr),c='m',marker='s')
ax3.errorbar(1,np.median(mdlx_corr),yerr=mdlx_err,c='m')
ax3.scatter(2,np.median(syn_corr),c='c',marker='s')  
ax3.errorbar(2,np.median(syn_corr),yerr=syn_err,c='c') 
ax3.set_xlim(0,3) 
ax3.set_ylim(0,0.5)
ax3.set_ylabel('Correlation strength')
ax3.set_xticks([1,2])
ax3.set_xticklabels(('Inhib','Excit'))

print('Mdlx correlation strength mean: {}'.format(np.nanmedian(mdlx_corr)))
#print('Mdlx SEM: {}'.format(mdlx_sem))
print('Mdlx IQR: {} - {}'.format(np.percentile(mdlx_corr,25),np.percentile(mdlx_corr,75)))
print('Syn correlation strength mean: {}'.format(np.nanmedian(syn_corr)))
print('Syn IQR: {} - {}'.format(np.percentile(syn_corr,25),np.percentile(syn_corr,75)))
stat,pval=ranksums(mdlx_corr,syn_corr)
print('Rank sum correlation strength: {:.3f}'.format(pval))
#%% Correlation Wavelength
ferrets=[21,60,143,150,181,182,183]
mdlx_spl=[]
for ferret in ferrets:
    wavelength=mdlx_data[ferret]['correlation_wavelength']
    mdlx_spl.append(wavelength) 
    
ferrets=[12,18,28,29,77,89,101]
syn_spl=[]
for ferret in ferrets:
    wavelength=syn_data[ferret]['correlation_wavelength']
    syn_spl.append(wavelength) 
    
    
mdlx_spl=np.array(mdlx_spl)
syn_spl=np.array(syn_spl)

mdlx_err=np.array([[np.nanmedian(mdlx_spl)-np.nanpercentile(mdlx_spl,25)],[np.nanpercentile(mdlx_spl,75)-np.nanmedian(mdlx_spl)]])
syn_err=np.array([[np.median(syn_spl)-np.percentile(syn_spl,25)],[np.percentile(syn_spl,75)-np.median(syn_spl)]])



# Now plot
ax1 = plt.subplot2grid(shape=(4, 5), loc=(2, 0), colspan=1)
ax2 = plt.subplot2grid((4, 5), (3, 0))
ax3 = plt.subplot2grid(shape=(4, 5), loc=(2, 1), colspan=1,rowspan=2)

# correlation spectrum
ferret=143
wavelength=mdlx_data[ferret]['correlation_wavelength']

if plotWavelengthSpectrum:
    wavelength=np.load(dataPath + '/F' + (f'{ferret:04}') + '_wavelength_spectrum.pkl',allow_pickle=True)
  
    # plotting subplots
    neighbourhood=wavelength['neighbourhood'];
    neighbourhood[np.isnan(neighbourhood)]=0
    ax1.imshow(neighbourhood,cmap=cm.RdBu_r,vmin=-0.5,vmax=0.5)
    ax1.axis('off')
    ax2.plot(wavelength['distance_mm'],wavelength['spectrum'],'k')
    ax2.set_xlim(0,1.5)
    ax2.set_xlabel('Distance (mm)')
    ax2.set_ylabel('Correlation')
#group wavelength
ax3.scatter(np.ones(len(mdlx_spl)) ,mdlx_spl,c='m',alpha=0.5)
ax3.scatter(np.ones(len(syn_spl)) * 2 ,syn_spl,c='c',alpha=0.5)
ax3.scatter(1,np.nanmedian(mdlx_spl),c='m',marker='s')
ax3.errorbar(1,np.nanmedian(mdlx_spl),yerr=mdlx_err,c='m')
ax3.scatter(2,np.nanmedian(syn_spl),c='c',marker='s')  
ax3.errorbar(2,np.nanmedian(syn_spl),yerr=syn_err,c='c') 
ax3.set_xlim(0,3) 
ax3.set_ylim(0.5,1.3)
ax3.set_ylabel('Wavelength (mm)')
ax3.set_xticks([1,2])
ax3.set_xticklabels(('Inhib','Excit'))

print('Mdlx correlation wavelength median: {}'.format(np.nanmedian(mdlx_spl)))
print('Mdlx IQR: {} - {}'.format(np.nanpercentile(mdlx_spl,25),np.nanpercentile(mdlx_spl,75)))
print('Syn correlation wavelength median: {}'.format(np.nanmedian(syn_spl)))
print('Syn IQR: {} - {}'.format(np.percentile(syn_spl,25),np.percentile(syn_spl,75)))
stat,pval=ranksums(mdlx_spl,syn_spl)
print('Rank sum correlation wavelength: {:.3f}'.format(pval))

#%% Eccentricity



mdlx_eccentricity=[]
ferrets=[21,60,143,150,181,182,183]
for ferret in ferrets:
    eccentricity=mdlx_data[ferret]['eccentricity']
    mn_eccentricity=np.nanmean(eccentricity)
    mdlx_eccentricity.append(mn_eccentricity)

syn_eccentricity=[]
ferrets=[12,18,28,29,77,89,101]
for ferret in ferrets:
    eccentricity=syn_data[ferret]['eccentricity']
    mn_eccentricity=np.nanmean(eccentricity)
    syn_eccentricity.append(mn_eccentricity)
    
mdlx_eccentricity=np.array(mdlx_eccentricity)
syn_eccentricity=np.array(syn_eccentricity)

mdlx_err=np.array([[np.median(mdlx_eccentricity)-np.percentile(mdlx_eccentricity,25)],[np.percentile(mdlx_eccentricity,75)-np.median(mdlx_eccentricity)]])
syn_err=np.array([[np.median(syn_eccentricity)-np.percentile(syn_eccentricity,25)],[np.percentile(syn_eccentricity,75)-np.median(syn_eccentricity)]])

fig.set_figheight(8)
fig.set_figwidth(10)  #4x6 for a 2x2 group subplot size
  

ax1 = plt.subplot2grid(shape=(4, 5), loc=(2, 2), colspan=1)
ax2 = plt.subplot2grid((4, 5), (3, 2))
ax3 = plt.subplot2grid(shape=(4, 5), loc=(2, 3), colspan=1,rowspan=2)

# eccentricity examples
if plotEccentricityExamples:
    ferret=143
    corr=mdlx_data[ferret]['correlation_matrix']
    roi=mdlx_data[ferret]['roi']
    ump=mdlx_data[ferret]['umperpixel']
    #calculate eccentricity of correlation patterns
    eccentricity,major,minor,cc_local_neighbourhood,ellipse_fit_params=correlationEccentricity(corr,roi,ump*4,region_size=500)

    # Now plot
    event_number='56'
    event_number='124'
    # plotting subplots
    ax1.imshow(cc_local_neighbourhood[int(event_number),:,:],interpolation='nearest',cmap='RdBu_r',vmin=-1,vmax=1)
    
    iellipse_params = ellipse_fit_params[event_number]["ellipse"]
    elle = Ellipse(xy=iellipse_params[0],width=iellipse_params[1][0],\
                						height=iellipse_params[1][1],angle=ellipse_fit_params[event_number]["rotation"])
    ax1.add_artist(elle)
    elle.set_facecolor('none')
    elle.set_edgecolor('k')
    elle.set_linewidth(3)
    ax1.set_title("Eccentricity={:.2f}".format(ellipse_fit_params[event_number]["eccentricity"]))
    ax1.axis('off')
    
    event_number='22'
    ax2.imshow(cc_local_neighbourhood[int(event_number),:,:],interpolation='nearest',cmap='RdBu_r',vmin=-1,vmax=1)
    
    iellipse_params = ellipse_fit_params[event_number]["ellipse"]
    elle = Ellipse(xy=iellipse_params[0],width=iellipse_params[1][0],\
                						height=iellipse_params[1][1],angle=ellipse_fit_params[event_number]["rotation"])
    ax2.add_artist(elle)
    elle.set_facecolor('none')
    elle.set_edgecolor('k')
    elle.set_linewidth(3)
    ax2.set_title("Eccentricity={:.2f}".format(ellipse_fit_params[event_number]["eccentricity"]))
    ax2.axis('off')

       
#group eccentricity
ax3.scatter(np.ones(len(mdlx_eccentricity)) ,mdlx_eccentricity,c='m',alpha=0.5)
ax3.scatter(np.ones(len(syn_eccentricity)) * 2 ,syn_eccentricity,c='c',alpha=0.5)
ax3.scatter(1,np.median(mdlx_eccentricity),c='m',marker='s')
ax3.errorbar(1,np.median(mdlx_eccentricity),yerr=mdlx_err,c='m')
ax3.scatter(2,np.median(syn_eccentricity),c='c',marker='s')  
ax3.errorbar(2,np.median(syn_eccentricity),yerr=syn_err,c='c') 
ax3.set_xlim(0,3) 
ax3.set_ylim(0.5,0.8)
ax3.set_ylabel('Eccentricity')
ax3.set_xticks([1,2])
ax3.set_xticklabels(('Inhib','Excit'))

print('Mdlx eccentricity median: {}'.format(np.nanmedian(mdlx_eccentricity)))
print('Mdlx IQR: {} - {}'.format(np.nanpercentile(mdlx_eccentricity,25),np.nanpercentile(mdlx_eccentricity,75)))
#print('Mdlx SEM: {}'.format(mdlx_sem))
print('Syn correlation eccentricity median: {}'.format(np.nanmedian(syn_eccentricity)))
print('Syn IQR: {} - {}'.format(np.percentile(syn_eccentricity,25),np.percentile(syn_eccentricity,75)))
#print('Syn SEM: {}'.format(syn_sem))
stat,pval=ranksums(mdlx_eccentricity,syn_eccentricity)
print('Rank sum eccentricity: {:.3f}'.format(pval))
#%% Fractures
ferrets=[21,60,143,150,181,182,183]
mdlx_mag=[]
for ferret in ferrets:
    mag=mdlx_data[ferret]['fracture_magnitude']
    mdlx_mag.append(mag)
    
ferrets=[12,18,28,29,77,89,101]
syn_mag=[]
for ferret in ferrets:
    mag=syn_data[ferret]['fracture_magnitude']
    syn_mag.append(mag)

mdlx_mag=np.array(mdlx_mag)
syn_mag=np.array(syn_mag)

mdlx_err=np.array([[np.median(mdlx_mag)-np.percentile(mdlx_mag,25)],[np.percentile(mdlx_mag,75)-np.median(mdlx_mag)]])
syn_err=np.array([[np.median(syn_mag)-np.percentile(syn_mag,25)],[np.percentile(syn_mag,75)-np.median(syn_mag)]])


fig.set_figheight(8)
fig.set_figwidth(10)  #4x6 for a 2x2 group subplot size
  

ax3 = plt.subplot2grid(shape=(4, 5), loc=(2, 4), colspan=1,rowspan=2)


ax3.scatter(np.ones(len(mdlx_mag)) ,mdlx_mag,c='m',alpha=0.5)
ax3.scatter(np.ones(len(syn_mag)) * 2 ,syn_mag,c='c',alpha=0.5)
ax3.scatter(1,np.median(mdlx_mag),c='m',marker='s')
ax3.errorbar(1,np.median(mdlx_mag),yerr=mdlx_err,c='m')
ax3.scatter(2,np.median(syn_mag),c='c',marker='s')  
ax3.errorbar(2,np.median(syn_mag),yerr=syn_err,c='c') 
ax3.set_xlim(0,3) 
ax3.set_ylim(0,0.005)
ax3.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax3.set_ylabel('Fracture magnitutde')
ax3.set_xticks([1,2])
ax3.set_xticklabels(('Inhib','Excit'))

print('Mdlx fracture magnitude median: {}'.format(np.nanmedian(mdlx_mag)))
print('Mdlx IQR: {} - {}'.format(np.nanpercentile(mdlx_mag,25),np.nanpercentile(mdlx_mag,75)))
#print('Mdlx SEM: {}'.format(mdlx_sem))
print('Syn fracture magnitude median: {}'.format(np.nanmedian(syn_mag)))
print('Syn IQR: {} - {}'.format(np.percentile(syn_mag,25),np.percentile(syn_mag,75)))
#print('Syn SEM: {}'.format(syn_sem))
stat,pval=ranksums(mdlx_mag,syn_mag)
print('Rank sum fracture magnitude: {:.3f}'.format(pval))
#%%