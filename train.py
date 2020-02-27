'''
    This file contains the training environment for DL4DI
    Misha Mesarcik 2019
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

from config import config
from preprocessor import preprocessor

import wandb
from wandb.keras import WandbCallback
from coolname import generate_slug
from sklearn.model_selection import train_test_split

import architecture
import visualisation
import analytic

from keras.datasets import mnist
import pickle


def get_analytics(model,X_train,X_test,y_train,y_test):
    """
        This function enables the retrieval of analytics data such as svm from the trained model
        model (keras.Model) the model from train.py
        X_train (np.array) the preprocessed data segmented into training data
        X_test (np.array) the preprocessed data segmented into test data
        y_train (np.array) the labels of the training data
        y_test (np.array) the labels of the test data

    """
    if y_train is not None:
        X_test = process(X_test)
        confusion_mx, report = analytic.get_analytics(model,X_train,X_test,y_train,y_test)
        wandb.log({"SVM Accuracy Report": wandb.Html( pd.DataFrame(report).to_html(), inject=False)})


def save_model(model):
    """
        This function saves the autoencoder (note that the encoder and decoder can be accessed by get_layer()
        Wandb is critical to the storage of these models

        autoencoder (keras.model): The model that gets saved to wandb
    """
    model.save(os.path.join(wandb.run.dir, config['name']))


def train(data):
    """
        This function gets an existing architecture specified by the input arguments and trains models
        according to the config file
    """
    model = architecture.get_architecture()
    wandb.init(project=config['project'],reinit=True,name=config['name'])
    config['loss_fn'] = config['loss_fn'].__name__
    wandb.config.update(config)

    if 'vae' in config['architecture']: y = None
    elif 'skip' in config['architecture']: y = None

    model.fit(x=data,
              y=y,
              epochs= config['n_epochs'],
              batch_size=config['batch_size'],
              callbacks=[WandbCallback()]
            )
    return model

def process(cube):
    """
        Uses the preprocessor class to process data before training.
        The application of which is defined by config.py

        cube (np.array): data in real/imag format with dimensions [baselines, freq, time, pol]
    """
    p = preprocessor(cube)
    p.interp(config['n_frequencies'],config['n_time_steps']) # always interpolate 
    
    # If architecture is skip then the apply a set preprocessing
    if config['architecture'] =='skip_mag_phase':
        p.get_phase()
        p.minmax(per_baseline=True,feature_range=(0,1))
        phase_cube = p.get_processed_cube()

        p = preprocessor(cube)
        p.interp(config['n_frequencies'],config['n_time_steps']) # always interpolate 
        p.get_magnitude()
        p.median_threshold()
        p.minmax(per_baseline=True,feature_range=(0,1))
        mag_cube = p.get_processed_cube()

        return [mag_cube,phase_cube]

    if config['architecture'] =='skip_real_imag':
        p = preprocessor(cube[...,0:1])
        p.interp(config['n_frequencies'],config['n_time_steps']) # always interpolate 
        p.median_threshold()
        p.minmax(per_baseline=True,feature_range=(0,1))
        real_cube = p.get_processed_cube()

        p = preprocessor(cube[...,1:])
        p.interp(config['n_frequencies'],config['n_time_steps']) # always interpolate 
        p.median_threshold()
        p.minmax(per_baseline=True,feature_range=(0,1))

        return [real_cube,p.get_processed_cube()]

    if config['architecture'] == 'vae_mag':
        p.get_magnitude()
        p.median_threshold()
        p.minmax(per_baseline=True,feature_range=(0,1))
        return p.get_processed_cube()

    if config['architecture'] == 'vae_phase':
        p.get_phase()
        p.minmax(per_baseline=True,feature_range=(0,1))
        return p.get_processed_cube()

    if config['architecture'] == 'vae_real':
        p = preprocessor(cube[...,0:1])
        p.interp(config['n_frequencies'],config['n_time_steps']) # always interpolate 
        p.median_threshold()
        p.minmax(per_baseline=True,feature_range=(0,1))
        return p.get_processed_cube()

    if config['architecture'] == 'vae_imag':
        p = preprocessor(cube[...,1:])
        p.interp(config['n_frequencies'],config['n_time_steps']) # always interpolate 
        p.median_threshold()
        p.minmax(per_baseline=True,feature_range=(0,1))
        return p.get_processed_cube()
    

def save_visualisations(autoencoder,data,labels,original_data):
    """
        Calls visualisation class to get the associated plots for each model  

        model (keras.Model) the keras autoencoder model to be visualised 
    """
    visualisation.get_visualsation(autoencoder,data,original_data,labels,wandb) 


def get_arguments():
    """
        Pretty self explanatory, gets arguments for training and adds them to config
    """
    parser = argparse.ArgumentParser(description='Train model for lofar-dev')
    parser.add_argument('training_data',metavar='-d', type=str, nargs = 1,
                        help = 'a dataset in the format [x_train,x_test,y_train,y_test]')
    parser.add_argument('architecture',metavar='-a', type=str, nargs = 1,
                        choices =['skip_mag_phase','skip_real_imag','vae_mag','vae_phase','vae_real','vae_imag'],
                        help = 'the architecture type like vae or ae_tnse')
    parser.add_argument('-latent_dim',metavar='-l', type=str, nargs = 1,
                        help = 'the dimension of the VAE embedding')
    parser.add_argument('-notes', metavar='-n', type=str, nargs = 1,
                        help = 'a filter for the clustering model to be visualised')
    parser.add_argument('-project', metavar='-p', type=str, nargs = 1,
                        help = 'The project name to be saved under in wandb')
    parser.add_argument('-wandb', metavar='-w', type=str, nargs = 1,
                        choices = [0,1],
                        help = 'Flag to set whether the wandb environment is used')
    args = parser.parse_args()

    config['architecture'] = args.architecture[0]
    config['training_data'] = args.training_data[0]
    config['latent_dim'] = int(args.latent_dim[0])
    config['name'] = generate_slug()

    if args.notes is not None:  config['Notes'] = args.notes[0]
    if args.wandb is not None: config['wandb'] = int(args.wandb[0])
    else:args.wandb = True 

    if args.project is not None: config['project'] = args.project[0]

def validate_config():
    """
        Does sanity checks for the config for training
    """
    assert (int(config['mag_phase']) +
            int(config['magnitude']) +
            int(config['phase']) +
            int(config['mag_fft']) +
            int(config['real_imag'])) == 1, ('The data can be exclusively in' 
                                        'the following formats: magnitude'
                                        'and phase, real and imaginary or only magnitude')
def get_data():
    """
        Gets the correct training data with train test split.
    """
    if config['Notes'] == "LOFAR":
        with open(config['training_data'],'rb') as f:
            x_train,_,labels,_ = pickle.load(f)

        x_train = np.concatenate([x_train[...,0:1],
                                  x_train[...,4:5]],axis=3)
        info = None

    else:
        with open(config['training_data'],'rb') as f:
            x_train,_,labels,_,info = pickle.load(f)

    if len(labels) > 1:
        X_train, X_test, y_train, y_test = train_test_split(x_train, labels, test_size = 0.20)
    else:
        X_train,X_test,y_train,y_test = x_train,None,None,None

    return  X_train,X_test,y_train,y_test, info
    
def main():
    '''
        Run all the important things
    '''
    get_arguments()
    validate_config()
    X_train,X_test,y_train,y_test,notes  = get_data()
    processsed_train_data = process(X_train)
    model = train(processsed_train_data)
    get_analytics(model,processsed_train_data,X_test,y_train,y_test)
    save_model(model)
    save_visualisations(model,processsed_train_data,y_train,X_train)

if __name__ == '__main__':
    main()
