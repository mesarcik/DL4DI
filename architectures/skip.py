"""
    2-input VAE adapated from the keras cvae documentation page found at https://keras.io/examples/variational_autoencoder/
    Misha Mesarcik 2020
"""

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda,BatchNormalization
from keras.layers import Reshape, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers import Multiply,Average,Maximum,Concatenate
from keras.layers import BatchNormalization 
from keras.models import Model
from keras.utils import plot_model
from keras import optimizers
from keras import backend as K
from keras.losses import binary_crossentropy
import os

import numpy as np
import sys
from coolname import generate_slug
import wandb
from wandb.keras import WandbCallback
from matplotlib import pyplot as plt
from random import randint

sys.path.insert(1,'../')
from preprocessor import preprocessor
from config import config

def get_skipped_vae(verbose=False):

    input_shape = (config['n_frequencies'], config['n_time_steps'],config['n_layers'])
    latent_dim = config['latent_dim']
    ####################################################################################
    ###############################    Make Encoder    #################################
    ####################################################################################
    phase_inputs = Input(shape=input_shape, name='phase_encoder_input')
    mag_inputs = Input(shape=input_shape, name='mag_encoder_input')
    mag,phase = mag_inputs,phase_inputs
    
    for layer in range(4):
        mag  = Conv2D(filters=config['filters'],
                   kernel_size=config['kernel_size'],
                   padding='same',
                   activation='relu',
                   name='mag_conv_{}'.format(layer))(mag)
        if layer != 3:
            mag = MaxPooling2D(pool_size =(2, 2),
                               padding='same',
                               name='mag_pool_{}'.format(layer))(mag)
        mag = BatchNormalization()(mag)

        phase = Conv2D(filters=config['filters'],
                   kernel_size=config['kernel_size'],
                   padding='same',
                   activation='relu',
                   name='phase_conv_{}'.format(layer))(phase)
        if layer != 3:
            phase = MaxPooling2D(pool_size =(2, 2),
                               padding='same',
                               name='phase_pool_{}'.format(layer))(phase)
        phase = BatchNormalization()(phase)

    ####################################################################################
    ###############################    Make Embedding  #################################
    ####################################################################################

    x = Concatenate()([mag,phase])
    x = BatchNormalization()(x)
    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(config['filters'], activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model([mag_inputs,phase_inputs], [z_mean, z_log_var, z], name='encoder')
    if verbose:
        encoder.summary()
        plot_model(encoder, to_file='outputs/vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)

    # TODO mayube need something here
    x = Reshape((shape[1], shape[2], shape[3]))(x)


    ####################################################################################
    ################################    Make Decoder   #################################
    ####################################################################################
    mag,phase = x,x
    for layer in range(3):
        mag  = Conv2DTranspose(config['filters'],
                            kernel_size=config['kernel_size'],
                            padding='same',
                            activation='relu',
                            name='mag_deconv_{}'.format(layer))(mag)
        mag = UpSampling2D(size=(2, 2),
                           name='mag_upsample_{}'.format(layer))(mag)
        mag = BatchNormalization()(mag)

        phase = Conv2DTranspose(config['filters'],
                            kernel_size=config['kernel_size'],
                            padding='same',
                            activation='relu',
                            name='phase_deconv_{}'.format(layer))(phase)
        phase = UpSampling2D(size=(2, 2),
                           name='phase_upsample_{}'.format(layer))(phase)
        phase = BatchNormalization()(phase)
        
    mag_outputs = Conv2DTranspose(config['n_layers'],
                     kernel_size=(2,2),
                     padding='same',
                     activation='sigmoid',
                     name = 'mag_output')(mag)
    phase_outputs = Conv2DTranspose(config['n_layers'],
                     kernel_size=(2,2),
                     padding='same',
                     activation='sigmoid',
                     name = 'phase_output')(phase)
    # instantiate decoder model
    mag_decoder =   Model(latent_inputs, mag_outputs, name='mag_decoder')
    phase_decoder = Model(latent_inputs, phase_outputs, name='phase_decoder')
    if verbose:
        mag_decoder.summary()
        phase_decoder.summary()
        plot_model(mag_decoder, to_file='outputs/vae_cnn_mag_decoder.png', show_shapes=True)
        plot_model(phase_decoder, to_file='outputs/vae_cnn_phase_decoder.png', show_shapes=True)

    # instantiate VAE model
    inputs = [mag_inputs,phase_inputs]
    outputs = [mag_decoder(encoder(inputs)[2]),
              phase_decoder(encoder(inputs)[2])]
    vae = Model(inputs, outputs, name='vae')

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = config['loss_fn'](K.flatten(inputs),
                                  K.flatten(outputs))

    reconstruction_loss *= config['n_frequencies']* config['n_time_steps'] 
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    adam = optimizers.adam(lr=0.0001)
    vae.compile(optimizer = adam)
    if verbose:
        vae.summary()
        plot_model(vae, to_file='outputs/vae_cnn.png', show_shapes=True)
    return vae


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
