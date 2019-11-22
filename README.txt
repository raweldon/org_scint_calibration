gamma_calibration.py
    - final code to implement gamma calibration
    - minimization results are pickled as a dataframe in gamma_cal_res.p (has full lmfit res object)
    - cpvert_bvert_cal.out is the output file from execution

uncert_gamma_cal.py
    - use to calculated uncertainties due to minimization
    - run after cleaning data (need lo for relative uncerts)
    - from plastic codes

cal_and_temp_changes.py
    - code to visualize changes in the calibration and temperature during the run
    - all pickles in this dir were made and used by this code

gamma_calibration_linear_interp.py
    - old version of gamma calibration with linear interpolation between each measurement

line_cal.py
    - calibration for just na-22 using a linear background term - works very well for both 343 and 1075 kev edges

linear_fit_vs_true.py
    - used to try and tell if calibration is not linear
   
