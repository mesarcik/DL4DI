import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import sys 
sys.path.insert(0, '/Users/mishamesarcik/Workspace/phd/Workspace/lofar-dev/')
from preprocessor import preprocessor

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}
matplotlib.rc('font', **font)

def plot_spectrograms(data,config):
    """
        This function makes a plot of 4 random spectrograms and their respective histograms
        cube (np.array) the processed data loaded in train.py
    """
    fig,axs = plt.subplots(4,7,figsize=(10,10))

    # Get all the correct cubes
    p = preprocessor(data)
    p.get_magnitude_and_phase()
    mag,phase = p.get_processed_cube()[...,0:1],p.get_processed_cube()[...,1:2]

    p = preprocessor(data)
    p.interp(config['n_frequencies'],config['n_time_steps'])
    p.get_magnitude_and_phase()
    mag_interp,phase_interp = p.get_processed_cube()[...,0:1],p.get_processed_cube()[...,1:2]

    p = preprocessor(data)
    p.interp(config['n_frequencies'],config['n_time_steps'])
    p.get_magnitude()
    p.median_threshold()
    p.minmax(per_baseline=True,feature_range=(np.min(phase_interp),np.max(phase_interp)))
    mag_interp_thresh = p.get_processed_cube()
    
    for i in range(4):
        r = np.random.randint(0,data.shape[0]) 
        im = axs[i,0].imshow(data[r,...,0]); 
        axs[i,0].title.set_text('Real Component')
        plt.colorbar(im,ax=axs[i,0])
        
        im = axs[i,1].imshow(data[r,...,1]); 
        axs[i,1].title.set_text('Imaginary Component')
        plt.colorbar(im,ax=axs[i,1])

        im = axs[i,2].imshow(mag[r,...,0]); 
        axs[i,2].title.set_text('Magnitude Component')
        plt.colorbar(im,ax=axs[i,2])

        im = axs[i,3].imshow(phase[r,...,0]); 
        axs[i,3].title.set_text('Phase Component')
        plt.colorbar(im,ax=axs[i,3])

        im = axs[i,4].imshow(mag_interp[r,...,0]); 
        axs[i,4].title.set_text('Magnitude component interpolated')
        plt.colorbar(im,ax=axs[i,4])

        im = axs[i,5].imshow(phase_interp[r,...,0]); 
        axs[i,5].title.set_text('Phase component interpolated')
        plt.colorbar(im,ax=axs[i,5])

        im = axs[i,6].imshow(mag_interp_thresh[r,...,0]); 
        axs[i,6].title.set_text('Magnitude component interpolated and thresholded')
        plt.colorbar(im,ax=axs[i,6])

    return plt    
