"""
    This is a class that wraps the preprocessing tasks, routines and objects
    Takes in a hypercube (N,M,P,C) dimensions where:
        N: baselines 
        M: frequency channels
        P: time vectors
        C: real and complex polarizations

    Misha Mesarcik 2020
"""
import copy
import logging
import os 
import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd 
from numpy import sqrt,angle
import pywt
from tqdm import tqdm

import processors 
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from skimage.transform import resize 


class preprocessor:
    def __init__(self, cube):
        self.cube= cube
        self.processed_flag = False
        self.processed_cube = np.array(None) 
        self.n_pol = cube.shape[-1]

        logging.basicConfig(filename='myapp.log', level=logging.INFO)

    def get_cube(self):
        return self.cube

    def _get_processed_flag(self):
        return self. processed_flag

    def _set_processed_flag(self,new_flag):
        self.processed_flag = new_flag 

    def get_correct_cube(self):
        if not self._get_processed_flag():
            self._set_processed_flag(True)
            return self.get_cube()
        else: 
            return self.get_processed_cube()

    def set_cube(self,cube):
        self.cube = cube

    def get_processed_cube(self):
        return self.processed_cube

    def set_processed_cube(self,processed_cube):
        self.processed_cube = processed_cube

    # TODO
    def wavelet_coef_shrinkage1D(self, 
                                 wavelet = 'haar',
                                 n = 3,
                                 threshold = 'fixed',
                                 tfix = 0.1,
                                 ttype = 'soft'):

        """
            This function applies the 1D wavelet transform with specified wavelet
            It then applies coefficient shrinkage using the specified threshold and type
            This is all done in parralell using joblib

            wavelet (str):      specifes which wavelet must to used, all wavelets in pywt are available
            n (int):            specifies number of levels in decomposition
            threshold (str):    specificies which threshold can be used (fixed, visushrink, )
            tfix (int):         specifies the threshold if the threshold parameter is fixed
            ttype (str):        specifies how the threshold should be applied (soft, hard)

        """
        logging.info('Wavelet Coefficient Shrinkage Applied')
        denoised_cube = self.get_correct_cube()
        
        a = Parallel(n_jobs=2)(
                                delayed(processors.coef_shrinkage_1D)(
                                       denoised_cube,
                                       baseline,
                                       channel,
                                       polarization,
                                       wavelet,
                                       n,
                                       threshold,
                                       tfix,
                                       ttype) 
                                       for baseline in range(self.cube.shape[0])
                                       for polarization in range(self.cube.shape[-1])
                                       for channel in range(self.cube.shape[1])
                                       )


        denoised_cube = np.reshape(a,denoised_cube.shape)
        self.set_processed_cube(denoised_cube)
    def wavelet_decomp_2D(self, 
                          wavelet = 'haar'):
        """ 
            This funcion uses the built in wavedec2 function from pywt to perform a 2d wavelet decomposition
            It the applies a concatenation function to regerate the same dimensions of the original image

            wavelet (str):      specifies wavelet name used for the decomposiont
        """
        logging.info('Wavelet 2D Decomposition Applied')

        decomp_cube = self.get_correct_cube()

        #Get new dimensions of cube
        temp_cube = decomp_cube[0,:,:,0]
        w_cube = pywt.dwt2(temp_cube, wavelet)
        a = processors.concat_2d_coefs(w_cube)
        w,l = a.shape 
        output_cube = np.zeros([decomp_cube.shape[0],w,l,decomp_cube.shape[-1]],dtype=complex)
        
        for baseline in range(decomp_cube.shape[0]):
            for polarization in range(decomp_cube.shape[-1]):
                im = decomp_cube[baseline,:,:,polarization]
                w = pywt.dwt2(im, wavelet)
                output_cube[baseline,:,:,polarization] = processors.concat_2d_coefs(w,verbose = False)
        
        self.set_processed_cube(output_cube)

    def median_threshold(self,per_baseline=True):
        """
            This function applies a threshold on the magnitude of the cube.
            The threshold is a multiple of the standard deviation

            sigma (int): the multiplication of the std threshold
            cube (np.array): the cube for the threshold to be applied to
        """
        logging.info('Sigma Threshold Applied')

        cube = self.get_correct_cube()
        thresholded_cube = copy.deepcopy(cube)

        if per_baseline:
            for i,baseline in enumerate(cube):
                median,std  = np.median(baseline),np.std(baseline)
                baseline  = np.where(baseline < median+std,baseline,median+std)
                baseline  = np.where(baseline > median-std,baseline,median-std)
                thresholded_cube[i,...] = baseline
        else:
            median,std  = np.median(cube),np.std(cube)
            thresholded_cube = np.where(thresholded_cube< median+std,thresholded_cube,median+std)
            thresholded_cube = np.where(thresholded_cube> median-std,thresholded_cube,median-std)
        
        self.set_processed_cube(thresholded_cube) 



    def mag2db(self):
        """
            This function converts the cube to a db scale and fills nans
            The conversion formula is: db = 20*log10(mag)

        """
        logging.info('Converting cube to db ')

        cube = self.get_correct_cube() 
        cube[cube==0] = 10**-4
        result = np.nan_to_num(20*np.log(cube))
        self.set_processed_cube(result) 


    # TODO
    def adaptive_filter(self):
        """
            This function applies an adaptive filter which functionality is wrapped in another file
        """
        logging.info('Adaptive Filtering Applied')

        adaptive_filtered_cube = []

        return adaptive_filtered_cube

    # TODO
    def flag(self,
            type = 'AOFlagger'):
        """
            This fucntion applies the a number of flagging techniques to each baseline

            type (str):     speicifies which flagging technique to be used current only AOFlagger
        """
        logging.info('Flagging Applied')

        flagged_cube = []

        return flagged_cube

    #TODO
    def mag_fft(self):
        """
            This function gets the 2d-fft of the input hypercube and get the magnitude

        """
        logging.info('Spectral Processing Applied')

        cube = self.get_correct_cube() 
        a,b,c,d = cube.shape
        new_cube = np.zeros([a,b,c,d])
        for i,baseline in enumerate(cube):
            for pol in range(d):
                new_cube[i,...,pol] = np.absolute(np.fft.fft2(baseline[...,pol]))

        self.set_processed_cube(new_cube) 

    def crop_cube(self,
                  crop_x=16,
                  crop_y=128):
        """
            This fucntion crops the hypercube to some x/y dimesion 

            x_crop (int):   The number of chanels to crop the hypercube to
            y_crop (int):   The number of time smaples to crop the hypercube to
        """
        cube = self.get_correct_cube()
        if (cube.shape[1] >= crop_x):
            start_x = np.random.randint(0, cube.shape[1] - crop_x + 1)
            end_x = start_x + crop_x
        else:
            start_x = 0
            end_x = cube.shape[1]

        if (cube.shape[2] >= crop_y):
            start_y = np.random.randint(0, cube.shape[2] - crop_y + 1)
            end_y = start_y + crop_y
        else:
            start_y = 0
            end_y = cube.shape[2]
            
        result = np.zeros((cube.shape[0], crop_x, crop_y, cube.shape[-1]))
        result[:, :cube.shape[1],:cube.shape[2],:] = cube[:, start_x: end_x, start_y: end_y, :]

        self.set_processed_cube(result)

    def interp(self,
               x_size ,
               y_size ,
               verbose = False):
        """
            This function resamples the existing hypercube to prevent the requirement of cropping
            x_size (int): the x dimenional size of the output_cube
            y_size (int): he y dimension size of the output cube
        """
        cube = self.get_correct_cube()
        if verbose:
            print('Interporlating hypercube')
            print('Original dimensions: {}'.format(cube.shape))

        output_cube = np.zeros([cube.shape[0],x_size,y_size,cube.shape[-1]])
        for baseline in range(cube.shape[0]):
            for polarization in range(cube.shape[-1]):
                im = cube[baseline,:,:,polarization]
                output_cube[baseline,:,:,polarization] = resize(im, [x_size,y_size],anti_aliasing=False)
        self.set_processed_cube(output_cube)
        if verbose: print('New dimensions: {}'.format(cube.shape))
    
    def standardise(self,per_baseline = True):
        """
            This function scales the hypercube (self.cube) using sklearn.standard scaler
            This is achieved on a baselines basis
        """
        scaled_cube = self.get_correct_cube() 
        if per_baseline:
            for i,baseline in tqdm(enumerate(scaled_cube),total= scaled_cube.shape[0]):
                scaled_cube[i,...] = (scaled_cube[i,...] - np.std(scaled_cube[i,...]))/np.mean(scaled_cube[i,...])


        else:
            scaled_cube = (scaled_cube-np.std(scaled_cube))/np.mean(scaled_cube)
            
        self.set_processed_cube(scaled_cube)


    def minmax(self,per_baseline = False,feature_range = (0,1)):
        """
            This function scales the hypercube (self.cube) using sklearn.MinMaxScaler scaler
            This is achieved on a baselines basis
        """
        scaled_cube = self.get_correct_cube()
        if per_baseline:
            for i,baseline in tqdm(enumerate(scaled_cube),total= scaled_cube.shape[0]):
                a,b,c = baseline.shape
                scaled_cube[i,...] = MinMaxScaler(feature_range=feature_range).fit_transform(
                                                  baseline.reshape([a*b,c])).reshape([a,b,c])
        else:
            a,b,c,d = scaled_cube.shape
            scaled_cube = MinMaxScaler(feature_range=feature_range).fit_transform(
                                        scaled_cube.reshape([a*b,c*d])).reshape([a,b,c,d])

        self.set_processed_cube(scaled_cube)

    def get_magnitude_and_phase(self):
        """
            This function returns the magnitude and phase associated with a particular cube
            The phase is in radians and the magnitude
        """
        cube = self.get_correct_cube()
        output_cube = np.zeros(cube.shape)
        if cube.shape[-1] ==1: 
            raise Exception('Cannot find phase of cube without imaginary component')

        else:
            mag = np.sqrt(cube[...,0:int(self.n_pol/2)]**2 + 
                          cube[...,int(self.n_pol/2):int(self.n_pol)]**2)

            phase = angle(cube[...,0:int(self.n_pol/2)] + 
                       1j*cube[...,int(self.n_pol/2):int(self.n_pol)])

            output_cube[...,0:int(self.n_pol/2)],output_cube[...,int(self.n_pol/2):int(self.n_pol)] = mag,phase

        self.set_processed_cube(output_cube)

    def get_magnitude(self):
        """
            This function returns the magnitude  associated with a particular cube
        """
        cube = self.get_correct_cube()
        output_cube = np.zeros(cube.shape)
        if cube.shape[-1] ==1: 
            output_cube = np.sqrt(cube**2)
            self.set_processed_cube(output_cube)
        else:
            mag = np.sqrt(cube[...,0:int(self.n_pol/2)]**2 + cube[...,int(self.n_pol/2):int(self.n_pol)]**2)

            self.set_processed_cube(mag)

    def get_phase(self):
        """
            This function returns the phase associated with a particular cube
            The phase is in radians 
        """
        cube = self.get_correct_cube()
        output_cube = np.zeros(cube.shape)
        if cube.shape[-1] ==1: 
            raise Exception('Cannot find phase of cube without imaginary component')

        else:
            mag = np.sqrt(cube[...,0:int(self.n_pol/2)]**2 + 
                          cube[...,int(self.n_pol/2):int(self.n_pol)]**2)

            phase = angle(cube[...,0:int(self.n_pol/2)] + 
                       1j*cube[...,int(self.n_pol/2):int(self.n_pol)])

            self.set_processed_cube(phase)

    def threshold(self,lower_bound = -0.5 ,upper_bound = 0.5):
        """
            This function thresholds in input cube between an upper and lower bound 
            lower_bound (int): The lower bound of the threshold 
            upper_bound (int): The  upper bound of the threshold
        """
        thresholded_cube = self.get_correct_cube() 
        upper_threshold_indices  = thresholded_cube < upper_bound
        lower_threshold_indices  = thresholded_cube > lower_bound

        indices = np.logical_and(lower_threshold_indices,
                                 upper_threshold_indices)

        thresholded_cube[indices] = 0

        self.set_processed_cube(thresholded_cube)


