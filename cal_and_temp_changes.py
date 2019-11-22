#!/usr/bin/env python
''' Code to analyze calibration measurements
    Uses gamma_calibration.py functions to fit spectra and calculate ADC value of 476 keV Cs-137 edge
        gives an idea of the change in the calibration with time
    function for plotting cs simulated and measured spectra - sim plots are shitty and not very helpful
'''    


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
import pandas as pd
from cal_and_temp_gamma_cal import simultaneous
import pickle
import os
from datetime import datetime
import sys
sys.path.insert(0, '/home/radians/raweldon/tunl.2018.1_analysis/') #give path to get_numpy_arrays
from get_numpy_arrays import get_numpy_arr, get_run_names

# directories
numpy_dir = '/media/radians/proraid/2018.1.tunl/beam_expt/out_numpy/'
sim_dir = '/home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/simulated_spectra/'
prefix = 'beam_expt_'
run_database = 'run_database.txt'

def temp_data():
    temp_data = pd.read_csv('stilbene_temp.csv')
    temp_data.columns = ['number', 'time', 'temp']

    run_temps = temp_data.iloc[110:2711]
    print '\nTemperature data:'
    print ' Avg (C)    Max (C)   Min (C)   Std'
    print '{:^8.2f} {:>8.1f} {:>8.1f} {:>8.2f}       all temps'.format(temp_data['temp'].mean(), temp_data['temp'].max(), 
                                                                       temp_data['temp'].min(), temp_data['temp'].std())
    print '{:^8.2f} {:>8.1f} {:>8.1f} {:>8.2f}       run temps'.format(run_temps['temp'].mean(), run_temps['temp'].max(), 
                                                                       run_temps['temp'].min(), run_temps['temp'].std())

    plt.figure()
    plt.plot(temp_data['number'], temp_data['temp'], 'o-', markersize=2, label='all temps')
    plt.plot(run_temps['number'], run_temps['temp'], 'o-', markersize=2, label='run temps')
    plt.legend()

def plot_cs(names, det_no, title, plot_cs_only):
    ''' set plot_cs_only = True if only plotting cs '''

    #print '\nPlotting cs spectra'
    block_print()
    #name = ('cs_bvert_11', 'cs_cpvert_11')
    #det_no = 2
    # flatten list
    names = [item for name in names for item in name]
    run = get_run_names(names, database_dir, run_database)
    plt.figure()
    colors = cm.viridis(np.linspace(0, 1, len(run)))
    for i, r in enumerate(run):
        if 'redo' in r:
            continue
        if plot_cs_only:
            if 'cs_' not in r:
                continue
        #print r
        numpy_arrs=[]
        data = get_numpy_arr(database_dir, run_database, r, numpy_dir, prefix, True)
        numpy_arrs.append(data)
        
        ql=[]
        for data_index, datum in enumerate(data):
            print data_index, datum
            f_in = np.load(numpy_dir + datum)
            data = f_in['data']
            ql_det = data['ql'][np.where((data['det_no'] == det_no))] # pat/plastic det 0, bvert det 1, cpvert det 2
            ql.extend(ql_det)
        
        plt.hist(ql, bins=1000, histtype='step', label=r, normed=True, color=colors[i])    
        plt.plot([0.477]*10, np.linspace(0, 4, 10), 'k--', linewidth=0.5, alpha=0.25) 
        if plot_cs_only:
            plt.xlim(0.3, 0.65)
            plt.ylim(0., 3.0)
        else:
            plt.xlim(0, 1.3)
            plt.title(title)
            plt.legend()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #plt.yscale('log')
        plt.xlabel('Energy deposited', fontsize=16)
        plt.ylabel('Normalized counts', fontsize=16)
        plt.tight_layout()
    plt.savefig('/home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/plots/' + title + '_cs_edges.pdf')
    enable_print()

    # plot sim data
    #sim_data = pickle.load( open('sim_with_res_' + title + '.p'))
    #plt.figure()
    #for i, sims in enumerate(sim_data):
    #    for idx, sim in enumerate(sims):
    #        sim_x, sim_y = sim
    #        sim_x_short = sim_x[np.where(sim_x > 0.25)]
    #        sim_y = sim_y[np.where(sim_x > 0.25)]
    #        sim_y_norm = sim_y/max(sim_y)
    #        plt.plot(sim_x_short, sim_y_norm, color=colors[i], label=run[i])
    #plt.plot([0.476]*10, np.linspace(0, 1, 10), 'k--', linewidth=0.5)
    #plt.ylim(0, 1)
    #plt.title(title)
    #plt.legend()

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

def plot_all():
    adc_476_11bvert, sig_x_11bvert = pickle.load( open('fit_params_11MeV_bvert.p'))
    adc_476_4bvert, sig_x_4bvert = pickle.load( open('fit_params_4MeV_bvert.p'))
    adc_476_11cpvert, sig_x_11cpvert= pickle.load( open('fit_params_11MeV_cpvert.p'))
    adc_476_4cpvert, sig_x_4cpvert= pickle.load( open('fit_params_4MeV_cpvert.p'))

    # print uncerts
    print '\n\nResults:'
    print '\n   crystal        sigma_avg  std_of_477_adc  max_min_%_diff  rel_std'
    print '{:^16s} {:>8.3f} {:>11.3f} {:>13.2f}% {:>12.2f}%'.format('11 MeV bvert:', np.mean(sig_x_11bvert), np.std(adc_476_11bvert), 
                                                    2*(max(adc_476_11bvert) - min(adc_476_11bvert))/(max(adc_476_11bvert) + min(adc_476_11bvert))*100,
                                                    np.std(adc_476_11bvert)/np.mean(adc_476_11bvert)*100)
    print '{:^16s} {:>8.3f} {:>11.3f} {:>13.2f}% {:>12.2f}%'.format('4 MeV bvert:',  np.mean(sig_x_4bvert), np.std(adc_476_4bvert), 
                                                    2*(max(adc_476_4bvert) - min(adc_476_4bvert))/(max(adc_476_4bvert) + min(adc_476_4bvert))*100,
                                                    np.std(adc_476_4bvert)/np.mean(adc_476_4bvert)*100)
    print '{:^16s} {:>8.3f} {:>11.3f} {:>13.2f}% {:>12.2f}%'.format('11 MeV cpvert:', np.mean(sig_x_11cpvert), np.std(adc_476_11cpvert),
                                                    2*(max(adc_476_11cpvert) - min(adc_476_11cpvert))/(max(adc_476_11cpvert) + min(adc_476_11cpvert))*100,
                                                    np.std(adc_476_11cpvert)/np.mean(adc_476_11cpvert)*100)
    print '{:^16s} {:>8.3f} {:>11.3f} {:>13.2f}% {:>12.2f}%'.format('4 MeV cpvert:', np.mean(sig_x_4cpvert), np.std(adc_476_4cpvert),
                                                    2*(max(adc_476_4cpvert) - min(adc_476_4cpvert))/(max(adc_476_4cpvert) + min(adc_476_4cpvert))*100,
                                                    np.std(adc_476_4cpvert)/np.mean(adc_476_4cpvert)*100)

    temp_data = pd.read_csv('stilbene_temp.csv')
    temp_data.columns = ['number', 'time', 'temp']
    run_temps = temp_data.iloc[110:2711]

    datetime_vals = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in run_temps['time'].values]
    dates = matplotlib.dates.date2num(datetime_vals)

    gs = gridspec.GridSpec(3,2)
    plt.figure(figsize=(12, 12))

    title_space = -0.3
    size = 14
    # temp
    ax = plt.subplot(gs[0, :])
    plt.plot(datetime_vals, run_temps['temp'], 'o-', markersize=2, linewidth=0.7)
    plt.ylabel('Temperature ($^{\circ}$C)', fontsize=size)
    plt.title('(a)', y=title_space, loc='left', fontsize=size)
    plt.gcf().autofmt_xdate()
    
    # 11 MeV bvert
    times = ('01-28 09:38', '01-28 18:18', '01-29 01:45', '01-29 17:26', '01-30 01:04', '01-30 08:23',
             '01-31 00:18', '01-31 07:38', '01-31 14:51', '01-31 22:13')
    ax = plt.subplot(gs[1, 0])
    plt.errorbar(range(len(adc_476_11bvert)), adc_476_11bvert, yerr=sig_x_11bvert, ecolor='black', markerfacecolor='None', fmt='o', 
                 markeredgecolor='r', markeredgewidth=1, markersize=7, capsize=1, linestyle='--', color='xkcd:gray', 
                 label='crystal 1, low gain mode')
    plt.plot(np.arange(0, len(adc_476_11bvert)), [np.mean(adc_476_11bvert) + np.std(adc_476_11bvert)]*len(adc_476_11bvert), 'b', alpha=0.25)
    plt.plot(np.arange(0, len(adc_476_11bvert)), [np.mean(adc_476_11bvert) - np.std(adc_476_11bvert)]*len(adc_476_11bvert), 'b', alpha=0.25)
    plt.plot(np.arange(0, len(adc_476_11bvert)), [np.mean(adc_476_11bvert)]*len(adc_476_11bvert), '--b', alpha=0.25)
    plt.fill_between(np.arange(0, len(adc_476_11bvert)),  [np.mean(adc_476_11bvert) + np.std(adc_476_11bvert)]*len(adc_476_11bvert), 
                     [np.mean(adc_476_11bvert) - np.std(adc_476_11bvert)]*len(adc_476_11bvert), color='C0', alpha=0.1)
    plt.ylabel('477 keV edge (ADC units)', fontsize=size)
    plt.xticks(np.arange(0, len(adc_476_11bvert)), times, rotation=30)
    plt.title('(b)', y=title_space, loc='left', fontsize=size)
    plt.legend()

    # 4 MeV bvert
    times = ('02-01 23:50', '02-02 06:30', '02-02 12:10', '02-02 17:27', '02-02 22:39', '02-03 07:35', '02-03 13:34', 
             '02-04 00:02', '02-04 05:36', '02-04 11:05', '02-04 21:20')
    ax = plt.subplot(gs[1, 1])
    plt.errorbar(range(len(adc_476_4bvert)), adc_476_4bvert, yerr=sig_x_4bvert, ecolor='black', markerfacecolor='None', fmt='o', 
                 markeredgecolor='r', markeredgewidth=1, markersize=7, capsize=1, linestyle='--', color='xkcd:gray', 
                 label='crystal 1, high gain mode')
    plt.plot(np.arange(0, len(adc_476_4bvert)), [np.mean(adc_476_4bvert) + np.std(adc_476_4bvert)]*len(adc_476_4bvert), 'b', alpha=0.25)
    plt.plot(np.arange(0, len(adc_476_4bvert)), [np.mean(adc_476_4bvert) - np.std(adc_476_4bvert)]*len(adc_476_4bvert), 'b', alpha=0.25)
    plt.plot(np.arange(0, len(adc_476_4bvert)), [np.mean(adc_476_4bvert)]*len(adc_476_4bvert), '--b', alpha=0.25)
    plt.fill_between(np.arange(0, len(adc_476_4bvert)),  [np.mean(adc_476_4bvert) + np.std(adc_476_4bvert)]*len(adc_476_4bvert), 
                     [np.mean(adc_476_4bvert) - np.std(adc_476_4bvert)]*len(adc_476_4bvert), color='C0', alpha=0.1)
                     
    plt.ylabel('477 keV edge (ADC units)', fontsize=size)
    plt.xticks(np.arange(0, len(adc_476_11bvert)), times, rotation=30)
    plt.title('(c)', y=title_space, loc='left', fontsize=size)
    plt.legend()
    
    # 11 MeV cpvert
    times = ('01-28 09:38', '01-28 18:18', '01-29 01:45', '01-29 17:26', '01-30 01:04', '01-30 08:23', '01-30 15:48', 
             '01-31 00:18', '01-31 14:51', '01-31 22:13')
    ax = plt.subplot(gs[2, 0])
    plt.errorbar(range(len(adc_476_11cpvert)), adc_476_11cpvert, yerr=sig_x_11cpvert, ecolor='black', markerfacecolor='None', fmt='o', 
                 markeredgecolor='r', markeredgewidth=1, markersize=7, capsize=1, linestyle='--', color='xkcd:gray', 
                 label='crystal 3, low gain mode')
    plt.plot(np.arange(0, len(adc_476_11cpvert)), [np.mean(adc_476_11cpvert) + np.std(adc_476_11cpvert)]*len(adc_476_11cpvert), 'b', alpha=0.25)
    plt.plot(np.arange(0, len(adc_476_11cpvert)), [np.mean(adc_476_11cpvert) - np.std(adc_476_11cpvert)]*len(adc_476_11cpvert), 'b', alpha=0.25)
    plt.plot(np.arange(0, len(adc_476_11cpvert)), [np.mean(adc_476_11cpvert)]*len(adc_476_11cpvert), '--b', alpha=0.25)
    plt.fill_between(np.arange(0, len(adc_476_11cpvert)), [np.mean(adc_476_11cpvert) + np.std(adc_476_11cpvert)]*len(adc_476_11cpvert), 
                     [np.mean(adc_476_11cpvert) - np.std(adc_476_11cpvert)]*len(adc_476_11cpvert), color='C0', alpha=0.1)
    #plt.xlabel('1-28-18 11:42 --- 1-31-18 06:00')
    plt.ylabel('477 keV edge (ADC units)', fontsize=size)
    plt.xticks(np.arange(0, len(adc_476_11bvert)), times, rotation=30)
    plt.title('(d)', y=title_space, loc='left', fontsize=size)
    plt.legend()
    
    # 4 MeV cpvert 
    times = ('02-01 23:50', '02-02 06:30', '02-02 12:10', '02-02 22:39','02-03 13:34', '02-0318:49', '02-04 00:02', 
             '02-04 05:36', '02-04 11:05', '02-04 16:13', '02-04 21:20')
    ax = plt.subplot(gs[2, 1])
    plt.errorbar(range(len(adc_476_4cpvert)), adc_476_4cpvert, yerr=sig_x_4cpvert, ecolor='black', markerfacecolor='None', fmt='o', 
                 markeredgecolor='r', markeredgewidth=1, markersize=7, capsize=1, linestyle='--', color='xkcd:gray', 
                 label='crystal 3, high gain mode')
    plt.plot(np.arange(0, len(adc_476_4cpvert)), [np.mean(adc_476_4cpvert) + np.std(adc_476_4cpvert)]*len(adc_476_4cpvert), 'b', alpha=0.25)
    plt.plot(np.arange(0, len(adc_476_4cpvert)), [np.mean(adc_476_4cpvert) - np.std(adc_476_4cpvert)]*len(adc_476_4cpvert), 'b', alpha=0.25)
    plt.plot(np.arange(0, len(adc_476_4cpvert)), [np.mean(adc_476_4cpvert)]*len(adc_476_4cpvert), '--b', alpha=0.25)
    plt.fill_between(np.arange(0, len(adc_476_4cpvert)), [np.mean(adc_476_4cpvert) + np.std(adc_476_4cpvert)]*len(adc_476_4cpvert), 
                     [np.mean(adc_476_4cpvert) - np.std(adc_476_4cpvert)]*len(adc_476_4cpvert), color='C0', alpha=0.1)
    #plt.xlabel('2-02-18 00:30 --- 2-04-18 22:00')
    plt.ylabel('477 keV edge (ADC units)', fontsize=size)
    plt.xticks(np.arange(0, len(adc_476_11bvert)), times, rotation=30)
    plt.legend()
    plt.title('(e)', y=title_space, loc='left', fontsize=size)
    plt.tight_layout(h_pad=2)
    plt.savefig('/home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/plots/temp_and_edges.pdf')

def main(beam_4mev, database_dir, det_no, exp_cal_term, names, spread, adc_ranges, f_name):
    ''' Note: comment out plt.show() in gamma_calibration.py if saving sim pickle '''

    sim_files = ['cs_spec_polimi.log','co_spec_polimi.log','na_spec_polimi.log']
    sim_numpy_files = ['cs_spec.npy', 'co_spec.npy', 'na_spec.npy'] # numpy files with data from sim_files

    if os.path.isfile('fit_params' + f_name + '.p'):
        print 'found fit_params' + f_name + '.p'
        adc_476, sig_x = pickle.load( open('fit_params' + f_name + '.p'))

    else:
        print '\n--------------' + f_name + '----------------'
        print '   m       m_std     b      b_std  correl_mb'
        sig_x, adc_476, sims_all = [], [], []
        for i, name in enumerate(names):
            run = get_run_names(name, database_dir, run_database)
    
            block_print()
            results, sims = simultaneous(adc_ranges, det_no, name, database_dir, sim_files, sim_numpy_files, run, exp_cal_term, 
                                         spread, beam_4mev, save_arrays=False, print_info=True, show_plots=True)
            enable_print()

            m, m_std, b, b_std, correl_mb = unpack_res(results[0])
        
            # uncert on LO
            print '{:^6.1f} {:>8.1f} {:>8.1f} {:>7.1f} {:>8.3f} '.format(m, m_std, b, b_std, correl_mb)
            lo_adc = 0.476*m + b
            sig = np.sqrt((0.476*m_std)**2 + b_std**2 + 2*0.476*correl_mb*b_std*m_std)
            sig_x.append(sig)
            adc_476.append(lo_adc)
            sims_all.append(sims[0])

        # save to pickle
        pickle.dump((adc_476, sig_x), open('fit_params' + f_name + '.p', 'wb'))
        pickle.dump(sims_all, open('sim_with_res' + f_name + '.p', 'wb'))

    plt.figure()
    plt.errorbar(range(len(adc_476)), adc_476, yerr=sig_x, ecolor='black', markerfacecolor='None', fmt='o', 
                 markeredgecolor='r', markeredgewidth=1, markersize=7, capsize=1, linestyle='--', color='xkcd:gray')

if __name__ == '__main__':
    # order of fitting 11MeV
    beam_4mev = False
    plot_cs_only = True

    database_dir = '/home/radians/tunl.2018.1/beam_expt/analysis/'
    names_bvert = (('cs_bvert_11MeV_0tilt', 'na_bvert_11MeV_0tilt', 'co_bvert_11MeV_0tilt'),
                   ('cs_cpvert_11MeV_0tilt', 'na_cpvert_11MeV_0tilt'),
                   ('cs_bvert_11MeV_45tilt', 'na_bvert_11MeV_45tilt'),
                   #('cs_bvert_11MeV_neg45tilt', 'na_bvert_11MeV_neg45tilt'),
                   ('cs_bvert_11MeV_30tilt', 'na_bvert_11MeV_30tilt'),
                   ('cs_bvert_11MeV_neg30tilt', 'na_bvert_11MeV_neg30tilt'),
                   ('cs_cpvert_11MeV_30tilt', 'na_cpvert_11MeV_30tilt'),
                   ('cs_bvert_11MeV_15tilt', 'na_bvert_11MeV_15tilt'),
                   ('cs_bvert_11MeV_neg15tilt', 'na_bvert_11MeV_neg15tilt'),
                   ('cs_cpvert_11MeV_15tilt', 'na_cpvert_11MeV_15tilt'),
                   ('cs_cpvert_11MeV_neg15tilt', 'na_cpvert_11MeV_neg15tilt'))

    names_cpvert = (('cs_bvert_11MeV_0tilt', 'na_bvert_11MeV_0tilt', 'co_bvert_11MeV_0tilt'), 
                    ('cs_cpvert_11MeV_0tilt', 'na_cpvert_11MeV_0tilt'), 
                    ('cs_bvert_11MeV_45tilt', 'na_bvert_11MeV_45tilt'),
                    ('cs_bvert_11MeV_30tilt', 'na_bvert_11MeV_30tilt'),
                    ('cs_bvert_11MeV_neg30tilt', 'na_bvert_11MeV_neg30tilt'), 
                    ('cs_cpvert_11MeV_30tilt', 'na_cpvert_11MeV_30tilt'),
                    ('cs_cpvert_11MeV_neg30tilt', 'na_cpvert_11MeV_neg30tilt'),
                    ('cs_bvert_11MeV_15tilt', 'na_bvert_11MeV_15tilt'),
                    ('cs_cpvert_11MeV_15tilt', 'na_cpvert_11MeV_15tilt'),
                    ('cs_cpvert_11MeV_neg15tilt', 'na_cpvert_11MeV_neg15tilt'))

    spread = 8000 # initial guess for spread term

    det_no = 1
    adc_ranges = ((2500, 4500), (6000, 9500), (6000, 11000)) # cs, na, co fit ranges
    exp_cal_term = 4020.0
    main(beam_4mev, database_dir, det_no, exp_cal_term, names_bvert, spread, adc_ranges, '_11MeV_bvert')
    plot_cs(names_bvert, det_no, '11MeV_bvert', plot_cs_only)

    det_no = 2
    adc_ranges = ((2500, 4500), (6000, 10500), (6000, 11000)) # cs, na, co fit ranges
    exp_cal_term = 4025.0
    main(beam_4mev, database_dir, det_no, exp_cal_term, names_cpvert, spread, adc_ranges, '_11MeV_cpvert')
    plot_cs(names_cpvert, det_no, '11MeV_cpvert', plot_cs_only)

    # 4MeV
    beam_4mev = True
    database_dir = '/home/radians/tunl.2018.1/beam_expt/analysis_4MeV/'
    names_bvert = (('cs_bvert_4MeV_0tilt', 'na_bvert_4MeV_0tilt'), 
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

    names_cpvert = (('cs_bvert_4MeV_0tilt', 'na_bvert_4MeV_0tilt'),
                    ('cs_cpvert_4MeV_0tilt', 'na_cpvert_4MeV_0tilt'),
                    ('cs_bvert_4MeV_45tilt', 'na_bvert_4MeV_45tilt'),
                    ('cs_bvert_4MeV_30tilt', 'na_bvert_4MeV_30tilt'),
                    ('cs_cpvert_4MeV_30tilt', 'na_cpvert_4MeV_30tilt'),
                    ('cs_cpvert_4MeV_neg30tilt', 'na_cpvert_4MeV_neg30tilt'),
                    ('cs_bvert_4MeV_15tilt', 'na_bvert_4MeV_15tilt'),
                    ('cs_cpvert_4MeV_15tilt', 'na_cpvert_4MeV_15tilt'),
                    ('cs_cpvert_4MeV_neg15tilt', 'na_cpvert_4MeV_neg15tilt'), 
                    ('cs_bvert_cpvert_end', 'na_bvert_cpvert_end', 'co_bvert_cpvert_end'))

    spread = 25000

    det_no = 1
    adc_ranges = ((7000, 14000), (19000, 30000), (15000, 31000))
    exp_cal_term = 11610.0
    main(beam_4mev, database_dir, det_no, exp_cal_term, names_bvert, spread, adc_ranges, '_4MeV_bvert')
    plot_cs(names_bvert, det_no, '4MeV_bvert', plot_cs_only)

    det_no = 2
    adc_ranges = ((7500, 14000), (19000, 30000), (16000, 31000))
    exp_cal_term = 11900.0
    main(beam_4mev, database_dir, det_no, exp_cal_term, names_cpvert, spread, adc_ranges, '_4MeV_cpvert')
    plot_cs(names_cpvert, det_no, '4MeV_cpvert', plot_cs_only)

    plot_all()
    temp_data()
    database_dir = '/home/radians/tunl.2018.1/beam_expt/analysis/'
    plt.show()

