#!/usr/bin/env python
''' final version of code to calculate the uncertainty due to the LO calibration
    calculates average uncertainties for each LO measured using the average between the start and end calibrations
'''
import numpy as np
import itertools
from gamma_calibration import simultaneous
import sys
sys.path.insert(0, '/home/radians/raweldon/tunl.2018.1_analysis/') #give path to get_numpy_arrays
from get_numpy_arrays import get_numpy_arr, get_run_names
import time
import pickle
import matplotlib.pyplot as plt
import os

start_time = time.time()

# directories
numpy_dir = '/media/radians/proraid/2018.1.tunl/beam_expt/out_numpy/'
sim_dir = '/home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/simulated_spectra/'
prefix = 'beam_expt_'
run_database = 'run_database.txt'

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def unpack_res(res):
    m = res.params['spread'].value
    m_std = res.params['spread'].stderr
    b = res.params['shift'].value
    b_std = res.params['shift'].stderr

    # get correlation between slope and intercetp from params correl
    m_correl = res.params['spread'].correl
    correl_mb = m_correl['shift']

    return m, m_std, b, b_std, correl_mb

def calc_avg_uncert(adc_ranges, det_no, names, database_dir, sim_files, sim_numpy_files, exp_cal_term, spread, beam_4mev):

    # need to update ql_means after calibration for accurate uncerts (updated 10/14/19)
    ql_means = ((1.534, 1.218, 0.841, 0.512, 0.253, 0.098), (5.072, 4.072, 2.893, 1.773, 0.874, 0.293))
                 
    angles = [70., 60., 50., 40., 30., 20.] 
    if beam_4mev:
        lo = ql_means[0]
        Ep = [4.825*np.sin(np.deg2rad(ang))**2 for ang in angles]
        print '4 MeV beam:'
    else:
        lo = ql_means[1]
        Ep = [11.325*np.sin(np.deg2rad(ang))**2 for ang in angles]
        print '11 MeV beam:'

    block_print()
    results = simultaneous(adc_ranges, det_no, names, database_dir, sim_files, sim_numpy_files,
                           exp_cal_term, spread, beam_4mev, print_info=False, show_plots=False)
    enable_print()

    sig_x = []; ms, m_stds, bs, b_stds, correl_mbs = [], [], [], [], []
    print '   m        m_std      b       b_std'
    print '-------------------------------------'
    for idx, res in enumerate(results):
        m, m_std, b, b_std, correl_mb = unpack_res(res)
        print '{:^8.2f} {:>8.2f} {:>8.2f} {:>8.2f}'.format(m, m_std, b, b_std)
        ms.append(m)
        bs.append(b)
        m_stds.append(m_std)
        b_stds.append(b_std)
        correl_mbs.append(correl_mb)

    # mean uncerts
    m = np.mean(ms)
    b = np.mean(bs)
    #m_std = np.sqrt(sum(x**2 for x in m_stds))/float(len(m_stds))
    #b_std = np.sqrt(sum(x**2 for x in b_stds))/float(len(b_stds))
    #m_std = np.mean(m_stds)
    #b_std = np.mean(b_stds)
    m_std = np.std(ms)
    b_std = np.std(bs)
    correl_mb = np.mean(correl_mbs)

    print '\nMean calibration values:'
    print '   m        m_std      b      b_std    correl_mb'
    print '---------------------------------------------------'
    print '{:^8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}\n'.format(m, m_std, b, b_std, correl_mb)

    lo_adc = np.array([l*m + b for l in lo])
    sig_x_avg = np.sqrt(((b - lo_adc)/m**2*m_std)**2 + (-b_std/m)**2 - 2*(b - lo_adc)/m**3*
                   (correl_mb)*b_std*m_std) # derived in 2/28/19 scrap notes
    
    print '  Ep    lo (MeVee)  sigma_lo (MeVee)    rel'
    print '---------------------------------------------'
    for w, x, y in zip(Ep, lo, sig_x_avg):
        print '{:^6.2f} {:>8.3f} {:>14.5f} {:>13.3f}% '.format(w, x, y, y/x*100)

def main(names, bvert, beam_4mev):
    ''' Provide beginning and end calibration files for a given detector to names list
    '''

    if beam_4mev:
        # 4mev
        spread = 25000

        if bvert:
            print '\ncalibration for bvert crystal (1) with 4 mev beam'
            det_no = 1
            exp_cal_term = 11610.0
            adc_ranges = ((7000, 14000), (19000, 30000), (15000, 31000))
        else:
            print '\ncalibration for cpvert crystal (2) with 4 mev beam'
            det_no = 2
            exp_cal_term = 11900.0
            adc_ranges = ((7500, 14000), (19000, 30000), (16000, 31000))
        database_dir = '/home/radians/tunl.2018.1/beam_expt/analysis_4MeV/'
    
    else:
        # 11mev
        spread = 8000 # initial guess for spread term
        
        if bvert:
            print '\ncalibration for bvert crsytal (1) with 11 mev beam\n' 
            det_no = 1
            exp_cal_term = 4020.0
            adc_ranges = ((2500, 4500), (6000, 9500), (6000, 11000)) # cs, na, co fit ranges
            names = ['cs_plastic_4MeV_0tilt', 'na_plastic_4MeV_0tilt', 'cs_bvert_4MeV_0tilt', 'na_bvert_4MeV_0tilt']
        else:
            print '\ncalibration for cpvert cyrstal (2) with 11 mev beam\n'
            det_no = 2
            exp_cal_term = 4025.0
            adc_ranges = ((2500, 4500), (6000, 10000), (6000, 11000)) # cs, na, co fit ranges
            names = ('cs_bvert_11MeV_0tilt', 'na_bvert_11MeV_0tilt', 'co_bvert_11MeV_0tilt', 'cs_cpvert_11MeV_neg15tiltend', 
                     'na_cpvert_11MeV_neg15tiltend')
        database_dir = '/home/radians/tunl.2018.1/beam_expt/analysis/'
 

    sim_files = ['cs_spec_polimi.log','co_spec_polimi.log','na_spec_polimi.log']
    sim_numpy_files = ['cs_spec.npy', 'co_spec.npy', 'na_spec.npy'] # numpy files with data from sim_files

    calc_avg_uncert(adc_ranges, det_no, names, database_dir, sim_files, sim_numpy_files, exp_cal_term, spread, beam_4mev)
    
if __name__ == '__main__':
    # file names
    names_bvert_4MeV = (('cs_bvert_4MeV_0tilt', 'na_bvert_4MeV_0tilt'), 
                        ('cs_cpvert_4MeV_0tilt', 'na_cpvert_4MeV_0tilt'),
                        ('cs_bvert_4MeV_45tilt', 'na_bvert_4MeV_45tilt'),
                        ('cs_bvert_4MeV_neg45tilt', 'na_bvert_4MeV_neg45tilt'),
                        ('cs_bvert_4MeV_30tilt', 'na_bvert_4MeV_30tilt'),
                        ('cs_bvert_4MeV_neg30tilt', 'na_bvert_4MeV_neg30tilt'),
                        ('cs_cpvert_4MeV_30tilt', 'na_cpvert_4MeV_30tilt'),
                        ('cs_bvert_4MeV_15tilt', 'na_bvert_4MeV_15tilt'),
                        ('cs_bvert_4MeV_neg15tilt', 'na_bvert_4MeV_neg15tilt'),
                        ('cs_bvert_4MeV_neg15tilt', 'na_bvert_4MeV_neg15tilt'),
                        ('cs_cpvert_4MeV_15tilt', 'na_cpvert_4MeV_15tilt'),
                        ('cs_bvert_cpvert_end', 'na_bvert_cpvert_end', 'co_bvert_cpvert_end'))
    
    names_cpvert_4MeV = (('cs_bvert_4MeV_0tilt', 'na_bvert_4MeV_0tilt'),
                         ('cs_cpvert_4MeV_0tilt', 'na_cpvert_4MeV_0tilt'),
                         ('cs_bvert_4MeV_45tilt', 'na_bvert_4MeV_45tilt'),
                         ('cs_bvert_4MeV_30tilt', 'na_bvert_4MeV_30tilt'),
                         ('cs_cpvert_4MeV_30tilt', 'na_cpvert_4MeV_30tilt'),
                         ('cs_cpvert_4MeV_neg30tilt', 'na_cpvert_4MeV_neg30tilt'),
                         ('cs_bvert_4MeV_15tilt', 'na_bvert_4MeV_15tilt'),
                         ('cs_cpvert_4MeV_15tilt', 'na_cpvert_4MeV_15tilt'),
                         ('cs_cpvert_4MeV_neg15tilt', 'na_cpvert_4MeV_neg15tilt'), 
                         ('cs_bvert_cpvert_end', 'na_bvert_cpvert_end', 'co_bvert_cpvert_end'))
    
    names_bvert_11MeV = (('cs_bvert_11MeV_0tilt', 'na_bvert_11MeV_0tilt', 'co_bvert_11MeV_0tilt'),
                         ('cs_cpvert_11MeV_0tilt', 'na_cpvert_11MeV_0tilt'),
                         ('cs_bvert_11MeV_45tilt', 'na_bvert_11MeV_45tilt'),
                         #('cs_bvert_11MeV_neg45tilt', 'na_bvert_11MeV_neg45tilt'), # bad, looks like Na-22 was too close to det when Cs was measured
                         ('cs_bvert_11MeV_30tilt', 'na_bvert_11MeV_30tilt'),
                         ('cs_bvert_11MeV_neg30tilt', 'na_bvert_11MeV_neg30tilt'),
                         ('cs_cpvert_11MeV_30tilt', 'na_cpvert_11MeV_30tilt'),
                         ('cs_bvert_11MeV_15tilt', 'na_bvert_11MeV_15tilt'),
                         ('cs_bvert_11MeV_neg15tilt', 'na_bvert_11MeV_neg15tilt'),
                         ('cs_cpvert_11MeV_15tilt', 'na_cpvert_11MeV_15tilt'),
                         ('cs_cpvert_11MeV_neg15tilt', 'na_cpvert_11MeV_neg15tilt'))
    
    names_cpvert_11MeV = (('cs_bvert_11MeV_0tilt', 'na_bvert_11MeV_0tilt', 'co_bvert_11MeV_0tilt'), 
                          ('cs_cpvert_11MeV_0tilt', 'na_cpvert_11MeV_0tilt'), 
                          ('cs_bvert_11MeV_45tilt', 'na_bvert_11MeV_45tilt'),
                          ('cs_bvert_11MeV_30tilt', 'na_bvert_11MeV_30tilt'),
                          ('cs_bvert_11MeV_neg30tilt', 'na_bvert_11MeV_neg30tilt'), 
                          ('cs_cpvert_11MeV_30tilt', 'na_cpvert_11MeV_30tilt'),
                          ('cs_cpvert_11MeV_neg30tilt', 'na_cpvert_11MeV_neg30tilt'),
                          ('cs_bvert_11MeV_15tilt', 'na_bvert_11MeV_15tilt'),
                          ('cs_cpvert_11MeV_15tilt', 'na_cpvert_11MeV_15tilt'),
                          ('cs_cpvert_11MeV_neg15tilt', 'na_cpvert_11MeV_neg15tilt'))

    main(names_bvert_4MeV, bvert=True, beam_4mev=True)
    main(names_cpvert_4MeV, bvert=False, beam_4mev=True)
    main(names_bvert_11MeV, bvert=True, beam_4mev=False)
    main(names_cpvert_11MeV, bvert=False, beam_4mev=False)


