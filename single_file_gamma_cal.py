#!/usr/bin/env python

''' Copy file from out_numpy to out_numpy/raw_cleaned before executing
'''

from gamma_calibration import remove_temp_cal
import pandas as pd
import numpy as np

numpy_dir = '/media/radians/proraid/2018.1.tunl/beam_expt/out_numpy/raw_cleaned/'

########## UPDATE FOR EACH FILE ################
file_name = 'beam_expt_2018_01_30-13_28_21_groomed_raw.npz'
det_no = 2
exp_cal_term = 4025.0
df = pd.read_pickle('gamma_cal_res.p')
df = df.xs(('11MeV', 'cpvert'))
##################################################

spread = np.mean(df['m'].values)
shift = np.mean(df['b'].values)

data = np.load(numpy_dir + file_name)
rec = data['data']
cal_det_index = np.where(rec['det_no'] == det_no)[0]
# remove rough calibration for qs and ql
rec['ql'][cal_det_index] = remove_temp_cal(rec['ql'][cal_det_index], exp_cal_term)
rec['qs'][cal_det_index] = remove_temp_cal(rec['qs'][cal_det_index], exp_cal_term)
# apply calibration for qs and ql
rec['ql'][cal_det_index] = (rec['ql'][cal_det_index] - shift)/spread
rec['qs'][cal_det_index] = (rec['qs'][cal_det_index] - shift)/spread
np.savez_compressed(numpy_dir + file_name, data=rec)
print '            ', file_name, 'saved'


