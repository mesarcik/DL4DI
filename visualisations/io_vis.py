"""
    This module is used to visualise the inputs and outputs of the vae or ae_tsne
    Misha Mesarcik 2019
"""
import numpy as np 
from matplotlib import image, pyplot as plt
import matplotlib
from random import sample,randint
import pandas as pd


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}
matplotlib.rc('font', **font)

def plot_io(autoencoder,data):
    """
        This function creates the plots of inputs and outputs of the autoencoder 
        autoencoder (keras.model) the autoencoder based model 
        data (np.array) the preprocessed training data that is in a list format. 
                            With the first index being magnitude and the second phase. 
    """
   
    output  = autoencoder.predict(data)
    titles = ['Magnitude','Phase']
    it = 1
    if isinstance(data,list): 
        it = 2
        shape = output[0].shape[0]
    else: 
        shape = output.shape[0]

    for i in range(it):
        fig,axs = plt.subplots(10,2,figsize=(10,10))

        if isinstance(data,list): 
            inp = data[i]
            outp = output[i]
        else: 
            inp = data
            outp = output

        for j in range(10):
            r = randint(0,shape-1)

            axs[j,0].imshow(inp[r,...,0]); 
            axs[j,0].title.set_text('Input baseline {}'.format(r))

            axs[j,1].imshow(outp[r,...,0]); 
            axs[j,1].title.set_text('Output baseline {}'.format(r))
        plt.suptitle('{} input/output pairs'.format(titles[i]))

    return plt
