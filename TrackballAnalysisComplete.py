# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:59:22 2022

@author: Laura Schröder
"""

# import packages

import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.signal #import savgol_filter
import seaborn as sns

plt.rcParams['figure.dpi'] = 300


## GET DATA LIST


## LOOP THROUGH EXPERIMENTS

    ## WHAT STIMULUS?
    
    # make a dataframe of zeros
    zeros = np.zeros(shape=(len(curr_events_raw), 5))
    curr_events = pd.DataFrame(zeros, columns={'timestamp', 'parameter','par_2', 'x_pos_change', 'disttr', 'rot_v'})
    
    # if gratings read data in like this
    for row in range(0, len(curr_events_raw)):
        #delete the bracket in current speed and save it as integer in new file
        
        curr_events.loc[:, 'speed'].iloc[row] = int(curr_events_raw.iloc[row, 0].replace('(', ''))

        # combine the last two values into the timestamp (were divided into 2 columns cause german computer
        # used comma as decimal seperator
        timestamp = str(curr_events_raw.iloc[row, 1]) + '.' + curr_events_raw.iloc[row, 2]

        # save timestamp as float in new file
        curr_events.loc[:, 'timestamp'].iloc[row] = float(timestamp.replace(')', ''))
        
    
    # if dots read it in like this
    for row in range(0, len(curr_events_raw)):
        #delete the bracket in current speed and save it as integer in new file
        curr_events.loc[:, 'speed'].iloc[row] = int(curr_events_raw.iloc[row,14].split("=",1)[1])
        curr_events.loc[:, 'dot_size'].iloc[row] = int(curr_events_raw.iloc[row,4].split("=",1)[1])
        curr_events.loc[:, 'dot_num'].iloc[row] = int(curr_events_raw.iloc[row,7].split("=",1)[1])
        
        # combine the last two values into the timestamp (were divided into 2 columns cause german computer
        # used comma as decimal seperator
        if isinstance(curr_events_raw.loc[row, 'Value'], float):
            timestamp = curr_events_raw.loc[row, 'Timestamp']
        else:
            timestamp = str(curr_events_raw.loc[row, 'Timestamp']) + '.' + curr_events_raw.loc[row, 'Value']

        # save timestamp as float in new file
        curr_events.loc[:, 'timestamp'].iloc[row] = float(timestamp.replace(')', ''))
    
    # space for different stimulus
    
    
    
    
    
    ## ANALYSIS
    
    
    
    ## PLOTS