"""
    A single input VAE adapted from the keras documentation found at https://keras.io/examples/variational_autoencoder/
    Misha Mesarcik 2020
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda,BatchNormalization
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import  MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K

from config import *

def get_vae(verbose=False):
    input_shape = (config['n_frequencies'], config['n_time_steps'],config['n_layers'])
    latent_dim = config['latent_dim'] 

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    ####################################################################################
    ###############################    Make Encoder    #################################
    ####################################################################################
    for layer in range(4):
        x = Conv2D(filters=config['filters'],
                   kernel_size=config['kernel_size'],
                   padding='same',
                   activation='relu',
                   name='conv_{}'.format(layer))(x)
        if layer != 3:
            x = MaxPooling2D(pool_size =(2, 2),
                               padding='same',
                               name='pool_{}'.format(layer))(x)
        x = BatchNormalization()(x)

    ####################################################################################
    ###############################    Make Embedding  #################################
    ####################################################################################

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
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    if verbose:
        encoder.summary()
        plot_model(encoder, to_file='outputs/vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    #x = Dense(config['filters'] * config['n_frequencies']/ 2 * config['n_time_steps']/ 2, activation='relu')


    # TODO mayube need something here
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    ####################################################################################
    ################################    Make Decoder   #################################
    ####################################################################################
    for layer in range(3):
        x = Conv2DTranspose(config['filters'],
                            kernel_size=config['kernel_size'],
                            padding='same',
                            activation='relu',
                            name='deconv_{}'.format(layer))(x)
        x = UpSampling2D(size=(2, 2),
                           name='upsample_{}'.format(layer))(x)
        x = BatchNormalization()(x)
        
    outputs = Conv2DTranspose(config['n_layers'],
                     kernel_size=(2,2),
                     padding='same',
                     activation='sigmoid',
                     name = 'output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    if verbose:
        decoder.summary()
    plot_model(decoder, to_file='outputs/vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # VAE loss = mse_loss or xent_loss + kl_loss
    #this was previously binary_crossentropy
    reconstruction_loss = config['loss_fn'](K.flatten(inputs),
                                  K.flatten(outputs))

    reconstruction_loss *= config['n_frequencies']* config['n_time_steps'] 
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer= config['optimizer'])
    if verbose:
        vae.summary()
        plot_model(vae, to_file='vae_cnn.png', show_shapes=True)
    return vae


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
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
