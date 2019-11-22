#!/usr/bin/env python
''' Used to compare calibrations
'''
import numpy as np
import matplotlib.pyplot as plt


def line(m, b, x):
    return m*x + b


def compare_m_b_change():
    ''' change in slope and intercept has large effect on low light output events
    '''
    # 4 MeV
    ms = 19448
    bs = -459
    
    # 11 MeV
    m = 6062
    b = -151
    
    a_adc = 2094.
    cp_adc = 1525.
    
    edges = (0.343, 0.476, 0.9635, 1.061, 1.3325)
    
    r = []
    for m, b in zip((5990, 6020, 6040), (-30., -90., -150.)):
        edges_adc = [e*m + b for e in edges]
        # ratios
        a = (a_adc - b)/m
        cp = (cp_adc - b)/m
        r.append(a/cp)
        xvals = np.linspace(0, 10.1, 100)
        plt.figure(0)
        plt.plot(edges, edges_adc, '-o')
        xvals = np.linspace(-0.03, 10.1, 100)
        plt.plot(xvals, [x*m + b for x in xvals], '--')
   
    print '\nchange in ratio with change in linear cal:'
    print '{:^8.2f} {:>8.2f} {:>8.2f} {:>8.2f}'.format(a_adc/cp_adc, r[0], r[1], r[2])
    
    plt.plot(xvals, [0]*len(xvals),  '-k')
    #plt.yscale('log')

def compare_line_cal_na_with_avg_cal():
    '''compares the slope and intercept of the average calibration terms used to analyze 
            stilbene data with the slope and intercept calculated fo na-22 using
            line_cal.py
       they are essentially identical...weird
    '''
    edges = (0.343, 0.476, 0.9635, 1.061, 1.3325)

    # true cal for bvert 11 MeV
    m = 8598.7
    b = -155.2
    edges_adc = [m*x + b for x in edges]
    
    na_343_m = 9171.4
    na_343_b = -343.6
    edge_343 = na_343_m*0.343 + na_343_b
    
    plt.figure()
    plt.plot(edges_adc, edges, '-o')
    xvals = np.linspace(-0.03, 10.1, 100)
    plt.plot([x*m + b for x in xvals], xvals, '--')
    plt.plot(edge_343, 0.343, 'o')
    
    print '\n343_edge_line 343_edge_avg   %diff'
    print edge_343, '    ', 0.343*m + b, '    ', 200*(edge_343-(0.343*m + b))/(edge_343+(0.343*m + b)), '%'
    
    # true cal for bvert 4 MeV
    m = 25868.35 
    b = -544
    edges_adc = [m*x + b for x in edges]
    
    na_343_m = 25717.
    na_343_b = -512.9
    edge_343 = na_343_m*0.343 + na_343_b
    
    plt.figure()
    plt.plot(edges_adc, edges, '-o')
    xvals = np.linspace(-0.03, 10.1, 100)
    plt.plot([x*m + b for x in xvals], xvals, '--')
    plt.plot(edge_343, 0.343, 'o')
    
    print edge_343, '    ', 0.343*m + b, '    ', 200*(edge_343-(0.343*m + b))/(edge_343+(0.343*m + b)), '%'

if __name__ == '__main__':

    compare_m_b_change()
    compare_line_cal_na_with_avg_cal()
    plt.show()


