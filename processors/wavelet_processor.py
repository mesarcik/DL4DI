import numpy as np

import pywt 
from pywt import wavedec

import matplotlib 
from matplotlib import pyplot as plt

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import compare_psnr

def decomp_vis(coefs, signal=None):
    """
        This function is used for visulization 1d wavelet decompositions

        coefs  (list of np arrays):      the decomposition coefficients from wavedec
        signal (optional 1d np array):   the original signal if desired to be plotted

    """
    if signal is not None:
        fig, ax = plt.subplots(len(coefs)+1, 1,constrained_layout =True)
        ax[0].plot(signal,c='orange');
        ax[0].set_title('Original Signal')
    else:
        fig, ax = plt.subplots(len(coefs), 1,constrained_layout =True)

    for i,coef in reversed(list(enumerate(coefs))):
        if signal is not None: ind = len(coefs) -i
        else: ind = i
        ax[ind].plot(coef)
        ax[ind].set_title('Decomposition Level ' + str(len(coefs) - i - 1) + ' Plot')

    plt.show()

def decomp_2vis(coefs, im = None):
    """
        This function is used for visulization 2d wavelet decompositions

        coefs  (list of np arrays):      the decomposition coefficients from wavedec
        im  (optional 2d np array):      the original image if desired to be plotted

    """
    if im is not None:
        fig, ax = plt.subplots(1, 2,constrained_layout =True)
        pos = ax[0].imshow(im);
        ax[0].set_title('Original Image')
        cbar = fig.colorbar(pos, ax=ax[0])

        im_ = concat_2d_coefs(coefs)
        pos = ax[1].imshow(im_ )
        #pos = ax[1].imshow(im_, vmin = cbar.vmin, vmax = cbar.vmax)
        ax[1].set_title('Multiresolution 2D tranformation')
        fig.colorbar(pos, ax=ax[1])

    else:
        plt.imshow(concat_2d_coefs(coefs))
        plt.imshow('Multiresolution 2D tranformation')
        plt.colorbar()
    print('show plot now ')
    plt.show()

def concat_2d_coefs(coefs,verbose = False, limit = 100):
    if verbose :
        print('The shape of coef[0]    = ' , coefs[0].shape) 
        print('The shape of coef[1][0] = ' , coefs[1][0].shape)
        print('The shape of coef[1][1] = ' , coefs[1][1].shape)
        print('The shape of coef[1][2] = ' , coefs[1][2].shape)

    output = np.concatenate([
             np.concatenate([coefs[0],    coefs[1][0]] ,axis =1 ),
             np.concatenate([coefs[1][1], coefs[1][2]] ,axis =1) ],
             axis = 0 )

    return output

def naive_denoising(im):
    """
    This funciton uses a naive approach (built in apporach) to denoising images
    The perfromance of this fucntion can be considerably improved with better selection of wavelet and trheshold
    """
    im_bayes = denoise_wavelet(im, multichannel=True,method='BayesShrink', mode='soft')
    return im_bayes

def coef_shrinkage_1D(cube, 
                      baseline,
                      channel,
                      polarization,
                      wavelet,
                      n,
                      threshold,
                      tfix,
                      ttype):
    """
        This fucntion performes 1D wavelet coefficient shrinkage based on a slice of a hypercube

        cube(np.array):     specifies the hyper cube to be acted on
        baseline (int):     specifies the baseline number to be acted on
        channel(int):       specifies the channel number to be acted on
        polarization(int):  specifices the polarization to be acted on
        wavelet (str):      specificies the wavelet type to be used in analysis and synthesis
        n (int):            specifies the DWT depth
        threshold(str):     specifies the type of threshold to be used
        tfix(int):          specifies the fixed threshold if to be used
        ttype(str):         specifies the type of thresholding to be applied (hard,soft)
    """
    slice = cube[baseline,channel,:,polarization]
    
    # Decomposition
    coefs = pywt.wavedec(slice, wavelet,level =n)

    # Theshold
    if threshold == 'fixed':
        denoised = coefs
        for i,coef in enumerate(coefs):
            denoised[i] = pywt.threshold(coef,tfix,ttype)
    else:
        logger.warning('No other wavelet thresholds have been impleted yet')
        return

    # Resynthesis
    return pywt.waverec(denoised,wavelet)[:slice.shape[0]]


if __name__ == "__main__":
    main()
