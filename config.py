"""
    The config file for all the required paths etc for the training and validation
    Misha Mesarcik 2020
"""
from keras.losses import binary_crossentropy

config ={
    #################################################
    #PATHS
    #################################################
    'path': '<USER_DEFINED>', 

    #################################################
    #MODEL PARAMS
    #################################################
    'architecture':'',
    'kernel_size': 3,
    'filters': 32,
    'n_layers':1,
    'stride':1,
    'optimizer': 'adam',
    'loss_fn': binary_crossentropy, 
    'avg_merge':True,
    'max_merge':False,
    'mult_merge':False,
    'latent_dim':2,

    #################################################
    #DATA PARAMS
    #################################################
    'training_data':'',
    'n_frequencies':32,
    'n_time_steps':128,
    'n_files': 30,
    'exclude':[],

    #################################################
    #PREPROCESSING PARAMS
    #################################################
    'mag_phase':True,
    'phase':False,
    'magnitude':False,
    'real_imag': False,
    'mag_fft':False,
    'median_threshold': False,
    'db':False,
    'minmax': True,
    'standardise': False,
    'per_baseline':False,
    'wavelets':False,
    'flag':False,
    'freq':False,
    'Notes': '', # capital 'n' because of wandb
    
    #################################################
    #TRAINING PARAMS
    #################################################
    'batch_size': 256,
    'n_epochs': 100,

    #################################################
    #WANDB PARAMS
    #################################################
    'name': '',
    'project': 'test',
}

