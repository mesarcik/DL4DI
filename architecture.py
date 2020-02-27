'''
    This file contains the architectural constraints of the models
'''
import architectures
from config import config

def get_architecture():
    """
        This returns the models defined in architecture as specified in config
    """

    if 'vae' in config['architecture']:
        print('getting vae')
        return architectures.get_vae()

    if 'skip' in config['architecture']:
        print('getting vae')
        return architectures.get_skipped_vae()
