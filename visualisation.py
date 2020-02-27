'''
    This file contains the calls to the visualisations  
    Misha Mesarcik 2020
'''
import visualisations
from config import config
import traceback


def get_visualsation(autoencoder,data,original_data,labels,wandb):
    """
        This returns the models defined in architecture as specified in config
    """
    assert (('vae' in config['architecture']) or 
            (config['architecture'] == 'ae') or
            ('skip' in config['architecture']) or
            (config['architecture'] == 'ae_tsne')), 'Architecture must be ae or vae'


    try:
        wandb.log({'spectrogram_analysis':visualisations.plot_spectrograms(original_data,
                                                                           config)})  
    except Exception as e:
        traceback.print_exc()
        pass
    try:
        wandb.log({'IO_analysis':visualisations.plot_io(autoencoder,
                                                        data)})  
    except Exception as e :
        traceback.print_exc()
        pass

    if 'vae' in config['architecture'] or 'skip' in  config['architecture']:
        try: 
           wandb.log({'vae_embedding':visualisations.plot_embedding(autoencoder,
                                                                    config)})  
        except Exception as e:
            traceback.print_exc()
            pass
        try: 
           wandb.log({'vae_overlay':(visualisations.plot_scatter(autoencoder,data))})  

        except Exception as e:
            traceback.print_exc()
            pass
        try: 
           wandb.log({'vae_scatter':wandb.Image(visualisations.plot_labeled_scatter(autoencoder,
                                                                          data,
                                                                          labels))})  
        except Exception as e:
            traceback.print_exc()
            pass
        try: 
           wandb.log({'vae_hist':wandb.Image(visualisations.plot_hist(autoencoder,
                                                               data))})  
        except Exception as e:
            traceback.print_exc()
            pass
