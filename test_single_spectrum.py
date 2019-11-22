#!/usr/bin/env python
''' updated version of simultaneous_fit.py 
    fits uncalibrated ADC spectra to simulation
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/radians/raweldon/tunl.2018.1_analysis/') #give path to get_numpy_arrays
from get_numpy_arrays import get_numpy_arr, get_run_names
import polimi_parse
import copy
from scipy.interpolate import interp1d
import lmfit
import os.path
import time

# directories
numpy_dir = '/media/radians/proraid/2018.1.tunl/beam_expt/out_numpy/'
sim_dir = '/home/radians/raweldon/tunl.2018.1_analysis/plastic_analysis/final_analysis/lo_calibration/simulated_spectra/'
prefix = 'beam_expt_'
run_database = 'run_database.txt'
start_time = time.time()

def get_sim_data(input_files):
    ''' Parse polimi log data.  Generates numpy arrays of simulation histograms.
        If npy files exist, they are loaded instead of the polimi log file.
    '''
    if os.path.exists(sim_dir+input_files[1]):
        print 'loaded '+input_files[1]
        return np.load(sim_dir+input_files[1])

    else:
        print 'Did not find spectrum numpy array.\nMaking array...'
        collision = polimi_parse.fromfile(sim_dir+input_files[0])
        histories = np.unique(collision.history)
        energy_deposition=[]

        for history in histories:
            #print history
            unique_hist_array = collision[np.where(collision.history == history)]
            delta_e = np.sum(unique_hist_array.deltaEnergy)
            energy_deposition.append(delta_e)
        energy_deposition = np.asarray(energy_deposition)

        max_data = 1.5
        bin_width = max_data/500. # changing this doesn't seem to make a difference (1/31/19)
        #make bin centers in the same x location for any spectrum
        sim_hist, sim_bin_edges = np.histogram(energy_deposition,
                                               bins=np.arange(0, max_data + bin_width, bin_width))
        sim_bin_centers = (sim_bin_edges[:-1] + sim_bin_edges[1:])/2
        sim_data = np.array((sim_bin_centers, sim_hist))
        np.save(sim_dir+input_files[1],sim_data) # save to npy for quick loading
        print 'spectra saved to ' + sim_dir+input_files[1]
        return sim_data
    
def remove_temp_cal(ql, exp_cal_term):
    ''' Undo rough calibration assigned during experiment
        m, b defined in /home/radians/mueller/analyzer/clean_and_calibrate_numpy_array.py 
    '''
    x1, y1, x2, y2 = 0, 0, exp_cal_term, 0.478 # values from ql_calibration.txt in expt analysis directory
    m = (y2 - y1)/(x2 - x1)  
    b = y1 - m*x1 
    uncalibrated_ql = [(q - b)/m for q in ql]
    return uncalibrated_ql

def get_meas_data(det_no, run_name, database_dir, exp_cal_term, beam_4_MeV):   
    numpy_arrs=[]
    data = get_numpy_arr(database_dir, run_database, run_name, numpy_dir, prefix, True)
    numpy_arrs.append(data)
    
    ql=[]
    for data_index,datum in enumerate(data):
        print data_index,datum
        f_in = np.load(numpy_dir + datum)
        data = f_in['data']
       
        scatter_index = np.where((data['det_no']==det_no))[0] # pat/plastic det 0, bvert det 1, cpvert det 2
        ql_det = remove_temp_cal(data['ql'][scatter_index], exp_cal_term)        
        ql.extend(ql_det)
    if beam_4_MeV:
        max_data = 35000
    else:        
        max_data = 12000
    bin_width = max_data/500.
    meas_hist, meas_bin_edges = np.histogram(ql,bins=np.arange(0, max_data + bin_width, bin_width))
    meas_bin_centers = (meas_bin_edges[:-1] + meas_bin_edges[1:])/2 
 
    meas_data = np.array((meas_bin_centers, meas_hist))

    #plt.figure()
    #plt.plot(meas_bin_centers, meas_hist)
    #plt.show()
    return meas_data    

def fit_range(low_bound, high_bound, data_in, name):
    '''select data for fit'''
    print '\nbounds:',name,low_bound,high_bound
    fit_range = (low_bound, high_bound)
    low_index = np.searchsorted(data_in[0], fit_range[0])
    high_index = np.searchsorted(data_in[0], fit_range[1])
    data_out = np.array((np.zeros(high_index-low_index), np.zeros(high_index-low_index)))
    data_out[0], data_out[1] = data_in[0][low_index:high_index], data_in[1][low_index:high_index]
    return data_out

def gaussian(x, mu, sigma, A):
    return A/np.sqrt(2.*np.pi*sigma**2) * np.exp(-(x - mu)**2/(2.*sigma**2))

def gaussian_smear(x, y, alpha, beta, gamma, c1, c2):
    '''smears bin centers (x) and bin contents (y) by a gaussian'''
    y_update = np.zeros(len(y))
    for i in xrange(len(x)):
        x_val = x[i]
        smear_val = np.sqrt((alpha*x_val)**2. + beta**2.*x_val + gamma**2.) 
        gaus_vals = gaussian(x, x_val, smear_val, 1)
        y_update[i] = np.dot(y, gaus_vals)
    return y_update
    
def shift_and_scale((alpha, beta, gamma, shift, spread, c1, c2, y_scale), bin_data):
    '''
    shifts and scales the guassian
    bin_data is an (x,y) numpy array of bin centers and values
    shift allows for a horizontal shift in the bin centers (e.g. bin centers (0.5, 1, 1.5) --> (1, 1.5, 2) )
    spread allows for horizontal scaling in the bin centers (e.g. bin centers (0.5, 1, 1.5) --> (1, 2, 3) )
    '''
    x, y = bin_data
    #y = c1**(-x + c2) + gaussian_smear(x, y, alpha, beta, gamma, c1, c2)
    y = gaussian_smear(x, y, alpha, beta, gamma, c1, c2)
    y *= y_scale
    x = copy.deepcopy(x)
    x_high = x[np.where(x > 0.04)]
    x_high = x_high*spread + np.ones(len(x_high))*shift
    x_low = x[np.where(x <= 0.04)]
    x_low = c1*x_low + c2*x_low**2
    x = np.concatenate((x_low, x_high))
    return x, y

def lin_scaling(shift, spread, c1, c2, bin_data):
    ''' scales uncalibrated measured data '''
    x, y = bin_data
    x = copy.deepcopy(x)
    x_high = x[np.where(x > 0.1)]
    x_high = (x_high - np.ones(len(x_high))*shift)/spread
    x_low = x[np.where(x <= 0.1)]
    x_low = (-c2 + np.sqrt(c2**2 - 4*c1*x_low))/2./c1
    print '+  ', (-c2 + np.sqrt(c2**2 - 4*c1*x_low))/2./c1
    print '-  ', (-c2 - np.sqrt(c2**2 - 4*c1*x_low))/2./c1
    return x, y
    
def calc_chisq(sim_x, sim_y, data_x, data_y):
    # bin centers in simulation are different from measurement -> use linear interpolation to compute SSE
    interp = interp1d(sim_x, sim_y, bounds_error=False, fill_value=0)
    sim_vals = interp(data_x)
    res = data_y - sim_vals
    chisq = [x**2/sim_vals[i] for i, x in enumerate(res)]  
    return chisq

def minimize(fit_params, *args):
    '''
    Minimization based on length of input arguements 
    sim_data is an (x, y) of the simulated bin centers and bin contents
    meas_data is an (x, y) of the measured bin centers and contents
    '''
    pars = fit_params.valuesdict()
    alpha_1 = pars['alpha_1']
    beta_1 = pars['beta_1']
    gamma_1 = pars['gamma_1']
    c1_1 = pars['c1_1']
    c2_1 = pars['c2_1']
    shift = pars['shift']
    spread = pars['spread']

    if len(args) in (1, 2, 3):
        #print 'minimizing 1'
        y_scale_1 = pars['y_scale_1']
        sim_data, meas_data = args[0]
        sim_x_1, sim_y_1 = shift_and_scale((alpha_1, beta_1, gamma_1, shift, spread, c1_1, c2_1, y_scale_1), sim_data)
        data_x_1, data_y_1 = meas_data
        chisq_1 = calc_chisq(sim_x_1, sim_y_1, data_x_1, data_y_1)     

    if len(args) in (2, 3):
        #print 'minimizing 2'
        alpha_2 = pars['alpha_2']
        beta_2= pars['beta_2']
        gamma_2= pars['gamma_2']
        c1_2 = pars['c1_2']
        c2_2 = pars['c2_2']
        y_scale_2 = pars['y_scale_2']
        sim_data, meas_data = args[1]
        sim_x_2, sim_y_2 = shift_and_scale((alpha_2, beta_2, gamma_2, shift, spread, c1_2, c2_2, y_scale_2), sim_data)
        data_x_2, data_y_2 = meas_data
        chisq_2 = calc_chisq(sim_x_2, sim_y_2, data_x_2, data_y_2)

    if len(args) == 3:
        #print 'minimizing 3'
        alpha_3 = pars['alpha_3']
        beta_3 = pars['beta_3']
        gamma_3= pars['gamma_3']
        c1_3 = pars['c1_3']
        c2_3 = pars['c2_3']
        y_scale_3 = pars['y_scale_3']
        sim_data, meas_data = args[2]
        sim_x_3, sim_y_3 = shift_and_scale((alpha_3, beta_3, gamma_3, shift, spread, c1_3, c2_3, y_scale_3), sim_data)
        data_x_3, data_y_3 = meas_data
        chisq_3 = calc_chisq(sim_x_3, sim_y_3, data_x_3, data_y_3)

    if len(args) == 1:
        #print chisq_1
        return chisq_1
    if len(args) == 2:
        #print chisq_1[0], chisq_2[0]
        return np.concatenate((chisq_1, chisq_2))
    if len(args) == 3: 
        #print chisq_1, chisq_2, chisq_3 
        return np.concatenate((chisq_1, chisq_2, chisq_3)) 
  
def spectra_fit(fit_params, *args, **kwargs):
    print '\nperforming minimization'

    fit_kws={'nan_policy': 'omit'}
    if len(args) in (1, 2, 3):
        sim_data_1, meas_data_full_1, meas_data_1 = args[0]
        if len(args) == 1:
            print '    single sprectrum fit'
            res = lmfit.minimize(minimize, fit_params, method='nelder', args=((sim_data_1, meas_data_1),), **fit_kws)

    if len(args) in (2, 3): 
        sim_data_2, meas_data_full_2, meas_data_2= args[1]
        if len(args) == 2:
            print '    double spectrum fit'
            res = lmfit.minimize(minimize, fit_params, args=((sim_data_1, meas_data_1), (sim_data_2, meas_data_2)), **fit_kws)

    if len(args) == 3:    
        sim_data_3, meas_data_full_3, meas_data_3= args[2]
        if len(args) == 3:
            print '    triple spectrum fit'
            res = lmfit.minimize(minimize, fit_params,
                                 args=((sim_data_1, meas_data_1), (sim_data_2, meas_data_2), (sim_data_3, meas_data_3)), 
                                 **fit_kws)

    if kwargs['print_info']:    
        print '\n',res.message
        print lmfit.fit_report(res)  
 
    if kwargs['show_plots']:
        if len(args) == 1:
            plot_fitted_spectra( res.params['shift'].value, res.params['spread'].value, 
                                  ( (res.params['alpha_1'].value,), (res.params['beta_1'].value,), (res.params['gamma_1'].value,),
                                    (res.params['c1_1'].value,), (res.params['c2_1'].value,),
                                    (res.params['y_scale_1'].value,), (sim_data_1,), (meas_data_full_1,), (meas_data_1,) ) )
        if len(args) == 2:
            plot_fitted_spectra( res.params['shift'].value, res.params['spread'].value, 
                                  ( (res.params['alpha_1'].value, res.params['alpha_2'].value), 
                                    (res.params['beta_1'].value, res.params['beta_2'].value), 
                                    (res.params['gamma_1'].value, res.params['gamma_2'].value),
                                    (res.params['c1_1'].value, res.params['c1_2'].value), (res.params['c2_1'].value, res.params['c2_2'].value),
                                    (res.params['y_scale_1'].value, res.params['y_scale_2'].value), (sim_data_1, sim_data_2),
                                    (meas_data_full_1, meas_data_full_2), (meas_data_1, meas_data_2) ) )
        if len(args) == 3:
            plot_fitted_spectra( res.params['shift'].value, res.params['spread'].value, 
                                  ( (res.params['alpha_1'].value, res.params['alpha_2'].value, res.params['alpha_3'].value), 
                                    (res.params['beta_1'].value, res.params['beta_2'].value, res.params['beta_3'].value), 
                                    (res.params['gamma_1'].value, res.params['gamma_2'].value, res.params['gamma_3'].value),
                                    (res.params['c1_1'].value, res.params['c1_2'].value, res.params['c1_3'].value), 
                                    (res.params['c2_1'].value, res.params['c2_2'].value, res.params['c2_3'].value),
                                    (res.params['y_scale_1'].value, res.params['y_scale_2'].value, res.params['y_scale_3']),
                                    (sim_data_1, sim_data_2, sim_data_3), (meas_data_full_1, meas_data_full_2, meas_data_full_3), 
                                    (meas_data_1, meas_data_2, meas_data_3) ) )
 
    # get shift term
    shift_term = res.params['shift'].value
    spread_term = res.params['spread'].value
    
    return shift_term, spread_term

def plot_fitted_spectra(shift, spread, args):
    for index, (alpha, beta, gamma, c1, c2, y_scale, sim_data, meas_data_full, meas_data) in enumerate(zip(args[0], args[1], 
                                                                                                           args[2], args[3], 
                                                                                                           args[4], args[5],
                                                                                                           args[6], args[7], args[8])):
        # update sim data
        sim_new = shift_and_scale((alpha, beta, gamma, shift, spread, c1, c2, y_scale), sim_data)
       
        # plot measured and fitted simulated data
        plt.figure(0)
        plt.plot(meas_data_full[0], meas_data_full[1], linestyle='None', marker='x', markersize=5, alpha=0.5, label='measured')
        plt.plot(sim_new[0], sim_new[1], '--', label='fit')
        #plt.plot([spread*x + shift for x in np.linspace(0, max(sim_data[0]), len(meas_data[0]))], 
        #         [c1*x + c2 for x in np.linspace(0, max(sim_data[0]), len(meas_data[0]))], '--')
        plt.xlabel('ql (MeVee)')
        plt.ylabel('Counts')
        plt.ylim(-100, 3000)
        plt.legend() 
         
        # update measured data and plot
        meas_scaled = lin_scaling(shift, spread, c1, c2, meas_data_full)
        sim_with_res = shift_and_scale((alpha, beta, gamma, 0, 1, c1, c2, y_scale), sim_data)

        plt.figure(1)            
        plt.plot(meas_scaled[0], meas_scaled[1], linestyle='None', marker='x', markersize=5, alpha=0.5, label='measured')
        plt.plot(sim_with_res[0], sim_with_res[1], '--', label='sim with res')
        plt.plot(sim_data[0], sim_data[1], alpha=0.3, label='sim data')      
 
        # plot edge locations
        if index == 0:
            a = 0.3
            y_min, y_max = plt.ylim()
            y_range = np.arange(0, 5000 + 100, 100.)
            plt.plot([0.478]*len(y_range), y_range, 'k--', alpha=a)
            plt.text(0.478, y_max - y_max/15, 'cs edge')
            plt.plot([0.343]*len(y_range), y_range, 'k--', alpha=a)
            plt.text(0.343, y_max, 'na edge')
            plt.plot([1.075]*len(y_range), y_range, 'k--', alpha=a)
            plt.text(1.075, y_max - y_max/10,'na edge')
            plt.plot([0.975]*len(y_range), y_range, 'k--', alpha=a)
            plt.text(0.975, y_max, 'co edge')
            plt.plot([1.133]*len(y_range), y_range, 'k--', alpha=a)
            plt.text(1.133, y_max - y_max/5, 'co edge')

        plt.ylim(1, 3000)
        plt.xlabel('ql (MeVee)')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.legend()

def cal_interp(shift_terms, spread_terms, start_file, stop_file, run_names, database_dir):
    ''' Interpolates between calibrations to get calibration for each measurement.
        Pass start_file (beginning calibration) and stop_file (end_calibration).
        returns numpy array with groomed numpy file name and calibration factor.
    '''
    print run_names
    run = get_run_names([run_names], database_dir, run_database) # all files to parse
    groomed_arrays = []
    print '\nFiles between start and stop:'
    for i,r in enumerate(run):
        print r
        cal_names = ('cs','na','co')
        if any(cal_name in r for cal_name in cal_names):
               continue
        groomed_arrays.extend(get_numpy_arr(database_dir, run_database, r, numpy_dir, prefix, True))

    num_files = float(len(groomed_arrays))
    print '\nshift_terms, spread_terms:', shift_terms, spread_terms

    if shift_terms[0] == shift_terms[1]: # accounts from end of 11MeV run with no end calibration
        #print [shift_terms[0]]*len(groomed_arrays)
        return [shift_terms[0]]*len(groomed_arrays), [spread_terms[0]]*len(groomed_arrays), groomed_arrays

    else:
        step_shift = (max(shift_terms) - min(shift_terms))/(num_files - 1) 
        interp_shift = np.arange(min(shift_terms), max(shift_terms) + step_shift/2, step_shift)

        # account for positive or negative change in shift parameter
        if shift_terms[0] > shift_terms[1]:
            interp_shift = interp_shift[::-1]

        step_spread = (max(spread_terms) - min(spread_terms))/(num_files - 1) 
        interp_spread = np.arange(min(spread_terms), max(spread_terms) + step_spread/2, step_spread)

        if spread_terms[0] > spread_terms[1]:
            interp_spread = interp_spread[::-1]

        print 'interp shift[0], shift[-1]:', interp_shift[0], interp_shift[-1] 
        print 'interp spread[0], spread[-1]:', interp_spread[0], interp_spread[-1]
        return interp_shift, interp_spread, groomed_arrays

def update_rec(shift, spread, groomed_files, det_no, exp_cal_term):
    print 'Saving to:', numpy_dir + 'raw_cleaned/'
    for i, g in enumerate(groomed_files):
        data = np.load(numpy_dir + g)
        rec = data['data']
        cal_det_index = np.where(rec['det_no'] == det_no)[0]
        # remove rough calibration
        rec['ql'][cal_det_index] = remove_temp_cal(rec['ql'][cal_det_index], exp_cal_term)
        # apply calibration
        rec['ql'][cal_det_index] = (rec['ql'][cal_det_index] - shift[i])/spread[i]
        rec['qs'][cal_det_index] = (rec['qs'][cal_det_index] - shift[i])/spread[i]
        tmp = g.split('.')
        name = tmp[:-1][0]
        np.savez_compressed(numpy_dir + 'raw_cleaned/' + name + '_raw', data=rec)
        print '    ', name, 'saved'
    return

def assign_fit_params(spread_val):
    # lmfit (curve fit wrapper, default: leastsq Levenberg-Marquardt)
    # Only fit beta (Kornilov does this)
    fit_params = lmfit.Parameters()
    fit_params.add('alpha_1', value=0.0, min=0., max=20., vary=False)
    fit_params.add('beta_1', value=0.04, min=0, max=0.05, vary=True)
    fit_params.add('gamma_1', value=0.0, min=0, max=20, vary=False)
    fit_params.add('alpha_2', value=0.0, min=0., max=20., vary=False)
    fit_params.add('beta_2', value=0.04, min=0, max=0.05, vary=True)
    fit_params.add('gamma_2', value=0.0, min=0, max=20, vary=False)
    fit_params.add('alpha_3', value=0.0, min=0., max=20., vary=False)
    fit_params.add('beta_3', value=0.04, min=0, max=0.05, vary=True)
    fit_params.add('gamma_3', value=0.0, min=0, max=20, vary=False)
    fit_params.add('shift', value=-100, vary=True)
    fit_params.add('spread', value=spread_val, min=0, vary=True)
    fit_params.add('c1_1', value=-100,  vary=True)
    fit_params.add('c2_1', value=100, vary=True)
    fit_params.add('c1_2', value=-100,  vary=True)
    fit_params.add('c2_2', value=100,  vary=True)
    fit_params.add('c1_3', value=-100,  vary=True)
    fit_params.add('c2_3', value=100,  vary=True)
    fit_params.add('y_scale_1', value=0.001, min=1e-5, max=1.0)
    return fit_params 

def simultaneous(det_no, names, database_dir, sim_files, sim_numpy_files, run, exp_cal_term, spread, beam_4_MeV, save_arrays):
    # group calibration run names 
    cal_names, cal_groups = [], []
    for r_index,r in enumerate(run):
        if r_index == 0:
            cal_names.append(r)
            continue
        tmp1 = r.split('_')
        if len(tmp1)>5:  # ensure correct format 
            continue
        tmp2 = run[r_index-1].split('_')
        if tmp1[1:] == tmp2[1:]:
            cal_names.append(r)
        else:
            cal_groups.append(cal_names)
            cal_names = []
            cal_names.append(r)
    cal_groups.append(cal_names)    

    if beam_4_MeV:
        cs_range = (7000, 10000)
        na_range = (18500, 23000)
        co_range = (18500, 24000)
    else:
        #cs_range = (2200, 3000)
        #na_range = (5700, 6500)
        #co_range = (5200, 6800)
        cs_range = (2000, 4000)
        na_range = (3000, 7500)
        co_range = (5000, 7500)


    shift_arr, spread_arr = [], []    
    for i, group in enumerate(cal_groups):
        print '\n---------------------------------------------------'
        print 'GROUP:'
        print  group
        if any('co' in g for g in group):
            co_cal = True
        else:
            co_cal = False
        for idx, g in enumerate(group):
            print '\n',g
            if 'cs_' in g:
                sim_data_cs = get_sim_data((sim_files[0],sim_numpy_files[0]))
                meas_data_full_cs = get_meas_data(det_no, g, database_dir, exp_cal_term, beam_4_MeV)
                meas_data_cs = fit_range(cs_range[0], cs_range[1], meas_data_full_cs,'meas_data_cs')  #0.35,0.7
                cs_data = [sim_data_cs, meas_data_full_cs, meas_data_cs]
                continue

            if 'na_' in g:
                sim_data_na = get_sim_data((sim_files[2],sim_numpy_files[2]))
                meas_data_full_na = get_meas_data(det_no, g, database_dir, exp_cal_term, beam_4_MeV)
                meas_data_na = fit_range(na_range[0], na_range[1], meas_data_full_na,'meas_data_na')  #0.95,1.5
                na_data = [sim_data_na, meas_data_full_na, meas_data_na]       
                fit_params = assign_fit_params(spread)       
                fit_params.add('y_scale_2', value=0.001, min=1e-5, max=0.1)        

            if 'co_' in g:
                # not using co for fits (results in poor fit with spread term - removed spread term)
                print 'found Co'
                sim_data_co = get_sim_data((sim_files[1],sim_numpy_files[1]))
                meas_data_full_co = get_meas_data(det_no, g, database_dir, exp_cal_term, beam_4_MeV)
                meas_data_co = fit_range(co_range[0], co_range[1], meas_data_full_co, 'meas_data_co')   #0.9,1.5
                co_data = [sim_data_co, meas_data_full_co, meas_data_co]
            
                # no na22 case
                if idx == 1:
                    fit_params = assign_fit_params(spread)
                    fit_params.add('y_scale_2', value=0.001, min=1e-5, max=0.1) 
                    shift, spread = spectra_fit(fit_params, cs_data, co_data, print_info=True, show_plots=True)
                else: 
                    fit_params.add('y_scale_3', value=0.001, min=1e-5, max=0.1)
                    shift, spread = spectra_fit(fit_params, cs_data, na_data, co_data, print_info=True, show_plots=True)

                    shift_arr.append(shift)
                    spread_arr.append(spread)
        
                    # interp only if fitting all 3 spectra
                    tmp = g.split('_')
                    if i == 0:
                        orientation = (tmp[1] +'_' + tmp[2] + '_' + tmp[3])
                    else:
                        interp_shift, interp_spread, groomed_arrays = cal_interp((shift_arr[i-1], shift_arr[i]), 
                                                                                 (spread_arr[i-1], spread_arr[i]),
                                                                                 cal_groups[i-1][:-1], g[0], orientation, database_dir)
                        orientation = (tmp[1] +'_' + tmp[2] + '_' + tmp[3])
                        plt.show()
                        if save_arrays:
                            update_rec(interp_shift, interp_spread, groomed_arrays, det_no, exp_cal_term) 
        
            if co_cal == False:
                shift, spread = spectra_fit(fit_params, cs_data, na_data, print_info=True, show_plots=True)

                # 4 MeV no co spectra
                shift_arr.append(shift)
                spread_arr.append(spread)
                
                tmp = g.split('_')
                if i == 0:
                    orientation = (tmp[1] +'_' + tmp[2] + '_' + tmp[3])
                else:
                    interp_shift, interp_spread, groomed_arrays = cal_interp((shift_arr[i-1], shift_arr[i]), 
                                                                             (spread_arr[i-1], spread_arr[i]),
                                                                             cal_groups[i-1][:-1], g[0], orientation, database_dir)
                    orientation = (tmp[1] +'_' + tmp[2] + '_' + tmp[3])
                    plt.show()
                    if save_arrays:
                        update_rec(interp_shift, interp_spread, groomed_arrays, det_no, exp_cal_term) 
 
        plt.show()

def single(det_no, names, database_dir, sim_files, sim_numpy_files, run, exp_cal_term, spread, beam_4_MeV):
    '''
    Use for individual spectrum fits to get rough idea of initial values and limits for simultaneous fitting
    Note: shift term is only accurate for simulataneous fit, do not use here
    '''
    for name in names:
        if 'cs_' in name:
            if beam_4_MeV:
                ranges = [7000, 10000]
            else:
                ranges = [2000, 3500]
            sim_file = sim_files[0]
            sim_numpy_file = sim_numpy_files[0]
            iso = 'cs'
        if 'na_' in name:
            if beam_4_MeV:
                ranges = [6000, 30000]
            else:
                ranges = [10, 9500] # 5200, 7000
            sim_file = sim_files[2]
            sim_numpy_file = sim_numpy_files[2]
            iso = 'na'
        if 'co' in name:
            if beam_4_MeV:
                ranges = [5000, 20000] # no co60 for 4 MeV
            else:
                ranges = [5000, 6800]
            sim_file = sim_files[1]
            sim_numpy_file = sim_numpy_files[1]
            iso = 'co'

        sim_data = get_sim_data((sim_file,sim_numpy_file))
        sim_data = [sim_data[0][20:], sim_data[1][20:]]
        
        meas_data_full = get_meas_data(det_no, name, database_dir, exp_cal_term, beam_4_MeV)
        meas_data = fit_range(ranges[0], ranges[1], meas_data_full, iso)  #0.35,0.7
        data = [sim_data, meas_data_full, meas_data]            

        # lmfit (curve fit wrapper, default: leastsq Levenberg-Marquardt)
        # Only fit beta (Kornilov does this)
        fit_params = lmfit.Parameters()
        fit_params.add('alpha_1', value=0.0, min=0., max=20., vary=False)
        fit_params.add('beta_1', value=0.04, min=0.035, max=0.1, vary=True)
        fit_params.add('gamma_1', value=0.0, min=0., max=20, vary=False)
        fit_params.add('shift', value=-100, max=0, vary=True)
        fit_params.add('spread', value=spread, min=0, vary=True) # 4mev 180000
        if 'co_' in name:
            fit_params.add('c1_1', value=-100, vary=True)
            fit_params.add('c2_1', value=100, vary=True)          
        else:            
            fit_params.add('c1_1', value=-100, vary=True)
            fit_params.add('c2_1', value=1000, vary=True)
        fit_params.add('y_scale_1', value=0.01)
         
        shift, spread = spectra_fit(fit_params, data, print_info=True, show_plots=True)
        plt.show()

if __name__ == '__main__':
    ''' 
    line background fit
    only use with na-22 
    '''
    det_no = 0  # 1 bvert, 2 cpvert, 0 plastic/pat
    beam_4mev = False
    if beam_4mev:
        database_dir = '/home/radians/tunl.2018.1/beam_expt/analysis_4MeV/'
        names = ['na_bvert_4MeV_0tilt']
        exp_cal_term = 11610.0 # from ql_calibration.txt in exp directory
        spread = 25000
    else:
        names = ['na_bvert_11MeV_0tilt']
        database_dir = '/home/radians/tunl.2018.1/beam_expt/analysis/'
        exp_cal_term = 4020.0
        spread = 8000

    sim_files = ['cs_spec_polimi.log','co_spec_polimi.log','na_spec_polimi.log']
    sim_numpy_files = ['cs_spec.npy', 'co_spec.npy', 'na_spec.npy'] # numpy files with data from sim_files
    run = get_run_names(names, database_dir, run_database)

    single(det_no, names, database_dir, sim_files, sim_numpy_files, run, exp_cal_term, spread, beam_4mev)
    #simultaneous(det_no, names, database_dir, sim_files, sim_numpy_files, run, exp_cal_term, spread, beam_4mev, save_arrays=False)
    print ("--- %s seconds ---" % (time.time() - start_time))
    plt.show()
