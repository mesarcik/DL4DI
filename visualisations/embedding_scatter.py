"""
    This module is used to visualise the embeddings of the VAE by superimposing the spectrograms over each point 
    Note this will not work with tnse as we need the 2 embedding of the autoencoder
    Misha Mesarcik 2019
"""
import numpy as np 
from random import sample,randint
import pandas as pd

import matplotlib
import matplotlib.image as mpimg
from matplotlib import image, pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import matplotlib.cm as cm
import traceback

import sys
sys.path.insert(0, '../')
from preprocessor import preprocessor

plt.rcParams['image.cmap'] = 'viridis'
titles = ['Magnitude','Phase']

def plot_scatter(autoencoder, data):
    """
        This function plots a 2d scatter plot with the data superimposed over each point
        autoencoder (keras.model) the autoencoder based model 
        data (np.array) the preprocessed training data that is in a list format. With the first index being magnitude and the second phase. 
    """
    plt.rcParams['image.cmap'] = 'viridis'
    encoder,mag_phase_flag  = load_encoder(autoencoder)

    if not mag_phase_flag:
        mag_data = data
        p = preprocessor(mag_data)
        it = 1
    else: 
        mag_data = data
        mag_data = [mag_data[0],
                    mag_data[1]]
        it = 2

    fig,ax = plt.subplots(1,it,figsize=(20,10));
    for i in range(it):
        p = preprocessor(mag_data[i])

        embeddings,_,_ =  encoder.predict(mag_data)
        p.interp(20,20)
        _data = p.get_processed_cube()

        for x, y, image_path in zip(embeddings[:,0], embeddings[:,1], _data[...,0]):
           imscatter(x, y, image_path, zoom=0.7, ax=ax[i]) 
        if it == 1: ax = [ax] # a hack to deal with single index
        ax[i].title.set_text(titles[i]);
        ax[i].grid();
        ax[i].set_xlim([-6,6])
        ax[i].set_ylim([-6,6])

    plt.suptitle('Scatter Plot of Embedding with Inputs Overlayed');
    plt.savefig('/tmp/temp.png',dpi=600)
    img=mpimg.imread('/tmp/temp.png')
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img) 
    plt.axis('off')
    return plt

def plot_labeled_scatter(autoencoder,  data,labels):
    """
        Overloaded function that plots the coloured scatter plot of the embedding 
        autoencoder (keras.model) the autoencoder based model 
        data (np.array) the preprocessed training data that is in a list format. With the first index being magnitude and the second phase. 
        labels (np.array) labels associated with data
    """
    encoder,mag_phase_flag  = load_encoder(autoencoder)
    if not mag_phase_flag:
        mag_data = data[0]
    else: mag_data = data

    embeddings,_,_ =  encoder.predict(data)

    colors = cm.rainbow(np.linspace(0, 1, len(pd.unique(labels))))
    fig,ax = plt.subplots(1,1,figsize=(10,10));
    lines = []
    for i,u in enumerate(pd.unique(labels)):
        l1 = ax.scatter(embeddings[:,0][labels == u],
                        embeddings[:,1][labels == u],
                        color=colors[i])

        lines.append(l1)
    ax.title.set_text('Scatter Plot of Embedding');#ax[0].grid();
    ax.legend(lines,list(pd.unique(labels)))
    ax.grid();
    plt.savefig('/tmp/temp.png',dpi=600)
    img=mpimg.imread('/tmp/temp.png')
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img) 
    plt.axis('off')
    return plt 

def plot_hist(autoencoder,data):
    """
        This function plots a 2d histogram showing the distribution of data 
        data (np.array) the preprocessed training data
        autoencoder (keras.model) the autoencoder based model 
    """
    plt.rcParams['image.cmap'] = 'jet'
    encoder,mag_phase_flag  = load_encoder(autoencoder)
    embeddings,_,_ =  encoder.predict(data)


    fig,ax = plt.subplots(1,1,figsize=(10,10));
    h,xedges,yedges,im = ax.hist2d(embeddings[:,0],
                                   embeddings[:,1],
                                   bins=150);
    ax.grid();
    plt.colorbar(im,ax=ax);
    ax.title.set_text('2D Histogram of Embedding');
    plt.savefig('/tmp/temp.png',dpi=600)
    img=mpimg.imread('/tmp/temp.png')
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img) 
    plt.axis('off')
    return plt 

def imscatter(x, y, image, ax=None, zoom=1):
    """
        code adapted from: https://gist.github.com/feeblefruits/20e7f98a4c6a47075c8bfce7c06749c2
    """
    if ax is None:
         ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def load_encoder(autoencoder):
    """
        Gets the encoders associated with the inputted model 
    """
    mag_phase_flag = False
    encoder = autoencoder.get_layer('encoder')
    dim = len(autoencoder.get_config()['input_layers'])

    return encoder, dim ==2
    
