'''
    This file contains the data generation environment for lofar-dev
    Misha Mesarcik 2019
'''
import numpy as np
import pickle
import datetime
from tqdm import tqdm
import os

import sys
sys.path.insert(1,'../')

import preprocessor
from h5_interface import *


def data_generator(num_files ):
    first_flag = False 
    ms_files = get_files(filter='None')

    for i in tqdm(range(0,num_files)):
        c = next(ms_files)
        cubes = get_cube(c)
        p = preprocessor.preprocessor(cubes)
        p.interp(32, 128)

        if not first_flag:
            output = p.get_processed_cube()
            first_flag = True
        else: output = np.concatenate((output,p.get_processed_cube()),axis=0)

    if not os.path.exists('datasets'):
        os.mkdir('datasets')

    info = {'Description':'LOFAR training set',
            'Features':'Unlabelled' ,
            'Dimensions':(32,128),
            'Source':'LOFAR MS'}


    f_name = 'datasets/LOFAR_dataset_{}.pkl'.format(datetime.datetime.now().strftime("%d-%m-%Y"))
    pickle.dump([output,np.zeros([1,1,1,1]), np.zeros([1,1,1,1]),np.zeros([1,1,1,1]),info],
            open(f_name, 'wb'), protocol=4)
    print('{} Saved!'.format(f_name))
def main():
    data_generator(2)

if __name__ == '__main__':
    main()

    
