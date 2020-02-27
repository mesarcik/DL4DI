"""
    This module is used to visualise the preprocessing of the data 
    Misha Mesarcik 2019
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import sys 
sys.path.insert(0, '/Users/mishamesarcik/Workspace/phd/Workspace/lofar-dev/')
from preprocessor import preprocessor
from config import config
from random import randint 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}
matplotlib.rc('font', **font)


def plot_preprocessing(data):
    fig,ax = plt.subplots(1,4)
    r = randint(0,data.shape[0])

    p = preprocessor(data)
    p.interp(config['n_frequencies'],config['n_time_steps']) # always interpolate 
    p.get_magnitude_and_phase() 

    d = p.get_cube()
    
    #Interpolated cube 
    im =ax[0].imshow(d[r,...,0])
    ax[0].title.set_text('Original Interpolated Image')
    fig.colorbar(im,ax=ax[0])

    #Magnitude of original cube 
    p.get_magnitude_and_phase() 
    p_cube = p.get_processed_cube()
    im =ax[1].imshow(p_cube[r,...,0])
    fig.colorbar(im,ax=ax[1])
    ax[1].title.set_text('Magnitude of Interpolated Image')


    p.sigma_threshold(2)

    # Standardised cube  
    p.standardise(per_baseline=config['per_baseline'])
    s_cube = p.get_processed_cube()
    im =ax[2].imshow(s_cube[r,...,0])
    ax[2].title.set_text('Standardised Interpolated Image')
    fig.colorbar(im,ax=ax[2])

    # Minmax scaled cube  
    p = preprocessor(p_cube)
    p.minmax(per_baseline=config['per_baseline'])
    m_cube = p.get_processed_cube()
    im =ax[3].imshow(m_cube[r,...,0])
    ax[3].title.set_text('Min Max Interpolated Image')
    fig.colorbar(im,ax=ax[3])
    
    return plt
