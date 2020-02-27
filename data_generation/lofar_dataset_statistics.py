'''
    This file saves a dataframe of the statistics associated with the training data
    Misha Mesarcik 2019
'''
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import os, os.path

import sys
sys.path.insert(1,'/home/mesarcik/lofar/phd/Workspace/lofar-dev/')

import preprocessor
from h5_interface import *


num_files = len([name for name in os.listdir('/home/mesarcik/lofar/data/')])
ms_files = get_files(filter='None')
df = pd.DataFrame(columns=['filename','baselines','frequency','time'])

for i in range(num_files):
    ms_file = next(ms_files)
    cube = get_cube(ms_file)
    data = [ms_file,cube.shape[0],cube.shape[1],cube.shape[2]]
    df.loc[i] = data

df.describe().to_csv('datasets/describe_log.csv')

df.to_csv('datasets/lofar.csv')


    



