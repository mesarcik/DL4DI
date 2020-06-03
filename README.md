[![DOI](https://zenodo.org/badge/242111858.svg)](https://zenodo.org/badge/latestdoi/242111858)


# `DL4DI`: Deep Learning Assisted Data Inspection for Radio Astronomy
A repository containing the implementation of the paper entitled "Deep Learning Assisted Data Inspection for Radio Astronomy"

## Installation 
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

For LOFAR dataset creation run the following from inside the data_generation directory given that the 'path' field is specified correctly in `config.py` and you have the correctly preprocessed `.h5` LOFAR spectrograms available. The downsampled dataset may be found at: https://doi.org/10.5281/zenodo.3702430. 

_Note that in order to use this dataset, each of the `.zip` files need to be extracted to the directory specified in 'path'_


``` 
    python3 generate_lofar_data.py
``` 

### Training 
Run the following given the correctly generated training files
```
    python3 train.py <training_file> <archtitecutre> -p <wandb_project> -l <latent_dim>
```

## Reference this work
```
@article{10.1093/mnras/staa1412,
    author = {Mesarcik, Michael and Boonstra, Albert-Jan and Meijer, Christiaan and Jansen, Walter and Ranguelova, Elena and van Nieuwpoort, Rob V},
    title = "{Deep Learning Assisted Data Inspection for Radio Astronomy}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2020},
    month = {05},
    issn = {0035-8711},
    doi = {10.1093/mnras/staa1412},
    url = {https://doi.org/10.1093/mnras/staa1412},
    note = {staa1412},
    eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/staa1412/33319604/staa1412.pdf},
}
```



## Notes
- The logging and visualisation data is dependant on wandb (https://www.wandb.com/)

## Licensing
Source code of DL4DI are licensed under the Apache License, version 2.0.

