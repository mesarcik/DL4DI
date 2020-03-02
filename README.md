# `DL4DI`: Deep learning for data inspection
A repository containing the implementation of the paper entitled "Deep Learning Assisted Data Inspection for Radio Astronomy"

## Installtion
Install conda environment by:
``` 
    conda create --name dl4di_test python=3.7
``` 
Run conda environment by:
``` 
    conda activate dl4di_test
``` 

Install dependancies by running:
``` 
    pip install -r requirements
``` 
## Usage

You need to create the training set using either `generate_hera_data.py` or `generate_lofar_data.py` (given you have access to the preprocessed lofar `.hdf5` files).

### Data set creation
For HERA data creation run the following from inside the data_generation directory
``` 
    python3 generate_hera_data.py
``` 

For LOFAR dataset creation run the following from inside the data_generation directory given that the 'path' field is specified correctly in `config.py` and you have the correctly preprocessed `.h5` LOFAR spectrograms available.

``` 
    python3 generate_lofar_data.py
``` 

### Training 
Run the following given the correctly generated training files
```
    python3 train.py <training_file> <archtitecutre> -p <wandb_project> -l <latent_dim>
```

## Notes
- The logging and visualisation data is dependant on wandb (https://www.wandb.com/)
