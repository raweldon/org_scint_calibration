#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sys
sys.path.insert(0, '/home/radians/raweldon/tunl.2018.1_analysis/') #give path to get_numpy_arrays
from get_numpy_arrays import get_numpy_arr, get_run_names

run_database = 'run_database.txt'
numpy_dir = '/media/radians/proraid/2018.1.tunl/beam_expt/out_numpy/'
prefix = 'beam_expt_'

def plot(database_dir, names, det_no):
    run = get_run_names(names, database_dir, run_database)
    plt.figure()
    colors = cm.viridis(np.linspace(0, 1, len(run)))
    for i, r in enumerate(run):
        if 'na_' in r:
            continue
        print r
        data = get_numpy_arr(database_dir, run_database, r, numpy_dir, prefix, True)
        
        ql=[]
        for data_index, datum in enumerate(data):
            print data_index, datum
            f_in = np.load(numpy_dir + datum)
            data = f_in['data']
            ql_det = data['ql'][np.where((data['det_no'] == det_no))] # pat/plastic det 0, bvert det 1, cpvert det 2
            ql.extend(ql_det)
        
        plt.hist(ql, bins=1000, histtype='step', label=r, normed=True, color=colors[i])    
        plt.plot([0.476]*10, np.linspace(0, 4, 10), 'k--', linewidth=0.5, alpha=0.25) 
        plt.xlim(0, 1.3)
        plt.title(r)
        plt.legend()

if __name__ == '__main__':
    names = (('cs_bvert_11MeV_0', 'cpvert_11MeV_0tilt', 'cs_cpvert_11MeV_neg15tilt'))
    #names = (('cpvert_11MeV_neg15tilt_355rot'),)
    database_dir = '/home/radians/tunl.2018.1/beam_expt/analysis/'
    plot(database_dir, names, det_no=1)
    plt.show()
