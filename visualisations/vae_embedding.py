"""
    Code to visualise vae embedding -- adapted from keras documentation
    Misha Mesarcik 2019
"""

import numpy as np 
from matplotlib import pyplot as plt
import matplotlib

import sys
sys.path.insert(0, '/Users/mishamesarcik/Workspace/phd/Workspace/lofar-dev/')
import preprocessor

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}
matplotlib.rc('font', **font)

def plot_embedding(autoencoder,config):
    """
        Plots the decoder outputs on a grid corresponding to the embedding
        In order to plot them, the decoder outputs are made square 

        autoencoder (keras.Model) The autoencoder to be plotted (must be a vae)
        config (dict) The configuration dictionary 
    """
    # TODO: Allow for multiple polarizations
    decoders,mag_phase_flag  = load_decoder(autoencoder)

    n = 20
    digit_size = np.max([config['n_frequencies'],config['n_time_steps']])
    figures = [np.zeros((digit_size * n, digit_size * n)),
               np.zeros((digit_size * n, digit_size * n))]

    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-10, 10, n)
    grid_y = np.linspace(-10, 10, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = []
            p = []
            digits = []
            if mag_phase_flag: # For mag and phase
                x_decoded.append(decoders[0].predict(z_sample))
                x_decoded.append(decoders[1].predict(z_sample))
                p.append(preprocessor.preprocessor(x_decoded[0][0].reshape([1,x_decoded[0][0].shape[0],
                                                   x_decoded[0][0].shape[1],
                                                   x_decoded[0][0].shape[2]])))
                p.append(preprocessor.preprocessor(x_decoded[1][0].reshape([1,x_decoded[1][0].shape[0],
                                                   x_decoded[1][0].shape[1],
                                                   x_decoded[1][0].shape[2]])))
                p[0].interp(digit_size,digit_size); p[0].get_magnitude()
                p[1].interp(digit_size,digit_size); p[1].get_magnitude()

                x_decoded[0] = p[0].get_processed_cube()[...,0]
                x_decoded[1] = p[1].get_processed_cube()[...,0]
                digits.append(x_decoded[0].reshape(digit_size, digit_size))
                digits.append(x_decoded[1].reshape(digit_size, digit_size))
                figures[0][i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digits[0]
                figures[1][i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digits[1]

            else: # For a single component
                x_decoded.append(decoders[0].predict(z_sample))
                p.append( preprocessor.preprocessor(x_decoded[0].reshape([1,x_decoded[0].shape[0],
                                                   x_decoded[0].shape[1],
                                                   x_decoded[0].shape[2]])))
                p[0].interp(digit_size,digit_size)
                p[0].get_magnitude()
                x_decoded[0] = p[0].get_processed_cube()[...,0]
                digits.append( x_decoded[0].reshape(digit_size, digit_size))
                figures[0][i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digits[0]

    fig,axs = plt.subplots(1,len(p),figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    titles = ['Magnitude','Phase']

    if len(p) == 1: # A hack to get indexing of suplots to work with only 1 subplot
        axs = [axs]
    for i,ax in enumerate(axs):
        ax.set_xticks(pixel_range)
        ax.set_xticklabels( sample_range_x)
        ax.set_yticks(pixel_range)
        ax.set_yticklabels(sample_range_y)

        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        im = ax.imshow(figures[i], cmap='viridis')
        ax.title.set_text('{} Embedding'.format(titles[i]))
        plt.colorbar(im,ax=ax)
    return  plt

def load_decoder(autoencoder):
    """
        Gets the decoders associated with the inputted model 
    """
    dim = len(autoencoder.get_config()['input_layers'])
    mag_phase_flag = False
    decoders = []
    if dim == 2: 
        mag_phase_flag = True
        decoders.append(autoencoder.get_layer('mag_decoder'))
        decoders.append(autoencoder.get_layer('phase_decoder'))
    else:
        decoders.append(autoencoder.get_layer('decoder'))
    return decoders,mag_phase_flag  
    
