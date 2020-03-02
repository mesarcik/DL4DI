"""
    This script generates a HERA training set for the DL4DI model. 
    It is adapted from the hera_sim documentations found at https://hera-sim.readthedocs.io/en/latest/tutorials/hera_sim_tour.html

    Misha Mesarcik 2020
"""

import aipy 
import numpy as np
import pandas as pd
import pylab as plt
from hera_sim import foregrounds, noise, sigchain, rfi
import datetime
import random
import pickle
from tqdm import tqdm
import copy
import os


feature_labels = {
          'noise':1,
          'point_source':1,
          'gains':0,
          'rfi_stations':0,
          'rfi_impulse':0,
          'rfi_dtv':0
          }
items = list(feature_labels.items())
label_list = [dict(items[:i]) for i in range(2,len(items)+1)]
class HERA_generator():

    def __init__(self):
        self.fqs = np.linspace(.1,.2,64,endpoint=False)
        self.lsts = np.linspace(0,2*np.pi,256, endpoint=False)
        self.times = self.lsts / (2*np.pi) * aipy.const.sidereal_day
        self.bl_len_ns_list = [5,10,20,100,500]
        self.n_sources_list = [1,10,50,200,1000]

        self.MX, self.DRNG = 2.5, 3
        self.Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        self.g = sigchain.gen_gains(self.fqs, [100,2,3])

        r = np.random.randint(0,5)
        self.bl_len_ns = self.bl_len_ns_list[r]
        #self.bl_len_ns = self.bl_len_ns_list[3]
        #self.n_sources = self.n_sources_list[np.random.randint(0,5)]
        self.n_sources = self.n_sources_list[3]

    def generate_diffuse_foreground(self):
        #GENERATE FOREGROUNDS
        return foregrounds.diffuse_foreground(self.lsts,self.fqs,self.bl_len_ns_list,Tsky_mdl=self.Tsky_mdl)/40


    def generate_point_source(self):
        #GENERATE Point-Source Foregrounds
        return foregrounds.pntsrc_foreground(self.lsts, self.fqs, self.bl_len_ns, nsrcs=self.n_sources) 
        
    def generate_noise(self):
        omega_p = noise.bm_poly_to_omega_p(self.fqs)
        tsky = noise.resample_Tsky(self.fqs,self.lsts,Tsky_mdl=noise.HERA_Tsky_mdl['xx'])
        t_rx = 150.
        return noise.sky_noise_jy(tsky + t_rx, self.fqs, self.lsts,omega_p)

    def generate_rfi_stations(self):
        return  rfi.rfi_stations(self.fqs, self.lsts)/200
    def generate_rfi_impulse(self):
        return rfi.rfi_impulse(self.fqs, self.lsts,strength=300, chance=.05)
    def generate_rfi_scatter(self):
        return rfi.rfi_scatter(self.fqs, self.lsts, strength=600,chance=.01)
    def generate_rfi_dtv(self):
        return rfi.rfi_dtv(self.fqs, self.lsts, strength=500,chance=.1)

    def generate_gains(self,vis):
        return sigchain.apply_gains(vis, self.g, (1,2))

    def generate_x_talk(self,vis):
        return sigchain.gen_cross_coupling_xtalk(self.fqs,vis,amp=1,dly=self.bl_len_ns)


    functions = [generate_diffuse_foreground,
                 generate_noise,
                 generate_point_source,
                 generate_rfi_stations,
                 generate_rfi_impulse,
                 generate_rfi_scatter,
                 generate_rfi_dtv,
                 generate_gains,
                 generate_x_talk
                 ]
                

def cointoss_addition(i,o):
    c = random.randint(0,1)
    if c:
        o +=i
    return o,c

def generate_label(labels):
    label_str = ''
    for key in labels:
        if labels[key] != 0:
            label_str = label_str + '-' + key
    return label_str[1:]


def generate_vis(n_features):
    hera = HERA_generator()

    labels = copy.deepcopy(label_list[n_features-2])
    output_vis = hera.generate_noise() 

    output_vis,label_flag = cointoss_addition(hera.generate_point_source(),output_vis)
    labels['point_source'] = label_flag

    if n_features >= 3:
        if random.randint(0,1):
            output_vis  = hera.generate_gains(output_vis) 
            labels['gains'] =1 

    if n_features >= 4:
        output_vis,label_flag = cointoss_addition(hera.generate_rfi_stations(),output_vis)
        labels['rfi_stations'] = label_flag

    if n_features >= 5:
        output_vis,label_flag = cointoss_addition(hera.generate_rfi_dtv(),output_vis)
        labels['rfi_dtv'] = label_flag

    if n_features >= 6:
        output_vis,label_flag = cointoss_addition(hera.generate_rfi_impulse(),output_vis)
        labels['rfi_impulse'] = label_flag



    return output_vis,generate_label(labels)

def make_dataset(n,n_features):
    data,label = [],[]

    for i in  tqdm(range(0,n)):
        g,l = generate_vis(n_features)
        data.append(np.array([g.real,g.imag]))
        label.append(l)
    
    return np.swapaxes(np.array(data),1,3),np.array(label)

def main():
    if not os.path.exists('datasets'):
        os.mkdir('datasets')

    for n_features in [5,2,3,4,6]:
        data,labels = make_dataset(10,n_features)
        
        info = {'Description':'Hera training set, with geometric baseline delay randomized',
                'Features':pd.unique(labels),
                'Dimensions':(64,256),
                'Source':'HERA Simulator'}

        f_name = 'datasets/HERA_{}_{}.pkl'.format(n_features,datetime.datetime.now().strftime("%d-%m-%Y"))
        pickle.dump([data,np.zeros([1,1,1,1]),labels, np.zeros([1,1,1,1]),info],
                open(f_name, 'wb'), protocol=4)
        print('{} Saved!'.format(f_name))
    

if __name__ =='__main__':
    main()


