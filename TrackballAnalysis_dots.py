# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 08:27:32 2022

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

# Set directory to data folder

# os.chdir("Z:\DATA\experiment_trackball_Optomotor")
os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data\Dots")

# get all folders in data (all animal IDs)

animals = os.listdir()
print(animals) # to see which animals

# loop through animals
for ID in animals:
    # go into animal folder
    os.chdir(ID)

    # get all days in animals folder
    days = os.listdir()


    # loop through days (currently only one per animal, but might change)
    
    #go into day folder
    

    # if os.listdir('data'):
    #     print('skip ' + ID + ' ' + day)
    #     os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data")
    #     continue


    # load trial event
    tr_ev = glob.glob('*.{}'.format('csv'))
    curr_events_raw = pd.read_csv(tr_ev[0])
    curr_events_raw=curr_events_raw.reset_index()


    # make a dataframe of zeros
    zeros = np.zeros(shape=(len(curr_events_raw), 7))
    curr_events = pd.DataFrame(zeros, columns={'timestamp', 'speed', 'dot_size', 'dot_num','x_pos_change', 'disttr', 'rot_v'})

    # loop through rows of event trials and save speed and timestamp to new dataframe
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

    # get dat file
    dat = glob.glob('*.{}'.format('dat'))
    curr_data_raw = pd.read_table(dat[0], sep="\s+")

    # create dataframe of zeros
    zeros = np.zeros(shape=(len(curr_data_raw), 7))
    curr_data = pd.DataFrame(zeros, columns={'timestamp', 'x_pos', 'y_pos', 'event', 'stim_speed', 'rot_v', 'dist'})

    # assign values to columns
    curr_data.loc[:, 'timestamp'] = curr_data_raw.iloc[:,24] # timestamp later used to match data to event

    curr_data.loc[:, 'x_pos'] = curr_data_raw.iloc[:, 14].replace({',': ''}, regex= True) # x position from fictrac data has a comma at end. has to be deleted
    curr_data.loc[:, 'y_pos'] = curr_data_raw.iloc[:, 15].replace({',': ''}, regex= True) # y position from fictrac data as a comma at end. has to be deleted
    curr_data.loc[:, 'x_pos'] = pd.to_numeric(curr_data.loc[:, 'x_pos']) # positions should be numeric (wasn't previously due to comma idk why it was there)
    curr_data.loc[:, 'y_pos'] = pd.to_numeric(curr_data.loc[:, 'y_pos'])

    curr_data.loc[:, 'rot_v'] = curr_data_raw.iloc[:, 7].replace({',': ''}, regex=True) # rotation around z axis
    curr_data.loc[:, 'rot_v'] = pd.to_numeric(curr_data.loc[:, 'rot_v']) * 144

    # apply filter to the raw trace
    curr_data.loc[:, ['x_pos']] = scipy.signal.savgol_filter(curr_data.loc[:, ['x_pos']], 99, 3, axis=0)
    curr_data.loc[:, ['y_pos']] = scipy.signal.savgol_filter(curr_data.loc[:, ['y_pos']], 99, 3, axis=0)
    # plt.subplot(222)
    # plt.plot(curr_data.loc[:, ['x_pos']], curr_data.loc[:, ['y_pos']])
    # # plt.plot(x_pos_19, y_pos_19)
    # # plt.subplot(223)
    # # plt.plot(x_pos_59, y_pos_59)
    # # plt.subplot(224)
    # # plt.plot(x_pos_99, y_pos_99)
    # plt.show()

    # calculate distance between consecutive points
    diff_dat = curr_data.loc[:, ['x_pos', 'y_pos']].diff()
    curr_data.loc[:, 'dist']= np.sqrt(diff_dat.loc[:,'x_pos'].pow(2) + diff_dat.loc[:,'y_pos'].pow(2))

    # gets the differences between timesteps of events
    #differences = np.ediff1d(curr_events.loc[:,'timestamp'])
    differences = curr_events.loc[:, 'timestamp'].diff()

    # the events are all supposed to be 30 seconds apart, so its divided by 6 to get the 5sec step
    five_steps = differences / 6

    five_step = np.mean(five_steps)

    # loops through events ( rows in event data)
    for ev_num in range(0, (len(curr_events.loc[:, 'timestamp'])-1)):
        #takes timestamp of current and next event
        event = curr_events.loc[ev_num, 'timestamp']
        next_event = curr_events.loc[ev_num+1, 'timestamp']

        # assigns current event number (ev_num +1 because i want it to start with 1 not 0) to event column of all datapoints that have a timestamp
        # bigger than 5 sec before stimulus start timestamp and smaller then next stim.start timestamp - 5 sec step
        curr_data.loc[(curr_data.loc[:, 'timestamp'] > (event-five_step)) & (curr_data.loc[:, 'timestamp'] < (next_event-five_step)), 'event'] = ev_num + 1
        curr_data.loc[(curr_data.loc[:, 'timestamp'] > event) & (curr_data.loc[:, 'timestamp'] < (next_event-five_step)), 'stim_speed'] = curr_events.loc[ev_num, 'speed']

        # ev dat is the x, y pos. and rotational velocity of current event when stimulus is present
        ev_dat = curr_data.loc[(curr_data.loc[:, 'event'] == (ev_num+1)) & (curr_data.loc[:, 'stim_speed'] != 0), ['x_pos', 'y_pos', 'rot_v', 'dist']]

        # calculates disttr within that event
        curr_events.loc[ev_num, 'disttr'] = np.nansum(ev_dat.loc[:, 'dist'])

        # calculates optomotor response index
        curr_events.loc[ev_num, 'rot_v'] = (np.nansum(abs(ev_dat.loc[ev_dat.loc[:,'rot_v']<0,'rot_v'])) - np.nansum(abs(ev_dat.loc[ev_dat.loc[:,'rot_v']>0,'rot_v'])))/(np.nansum(abs(ev_dat.loc[ev_dat.loc[:,'rot_v']<0,'rot_v'])) + np.nansum(abs(ev_dat.loc[ev_dat.loc[:,'rot_v']>0,'rot_v'])))

    # # Plots to figure out threshold for events ( decided on 1)
    # plt.hist(curr_events.loc[:, 'disttr'], bins = 20)
    # plt.xlabel('distance travelled')
    # plt.show()

    # print('percent of events under treshold')
    # print(((len(curr_events.loc[curr_events.loc[:, 'disttr'] <= 1, ['speed', 'disttr']]))/120)*100)
    # curr_events = curr_events.loc[curr_events.loc[:, 'disttr'] > 1, :]
    # curr_events.loc[curr_events.loc[:,'speed']<0,'dot_size'] =  curr_events.loc[curr_events.loc[:,'speed']<0,'dot_size'] * -1
   
    # plt.scatter(curr_events.loc[:, 'dot_size'], curr_events.loc[:, 'rot_v'])

    # plt.show()

    # # takes the rotation vectors of all 4 and 8 Hz events and plots them
    # eight_four = curr_events.loc[(curr_events.loc[:, 'speed'] >= 4) | (curr_events.loc[:, 'speed'] <= -4), ['disttr', 'rot_v']]

    # plt.scatter(np.arange(1,len(eight_four)+1, 1), eight_four.loc[:, 'rot_v'].abs())
    # plt.xlabel('event number')
    # plt.ylabel(' absolute Optomotor response index')
    # plt.show()
    
    # plt.scatter(np.arange(1, len(eight_four) + 1, 1), eight_four.loc[:, 'disttr'].abs())
    # plt.xlabel('event number')
    # plt.ylabel('distance travelled')
    # plt.show()
 

    # saves data so this doesnt have to be run for every future plot
    # os.chdir('data')
    # curr_data.to_csv('datatest.csv')
    # curr_events.to_csv('evtest.csv')

    # calculate accumulated rotation for each event. take average for each frame
    # save in dataframe
    
    # four_event = curr_data.loc[curr_data.loc[:,'stim_speed']== 4, ['event', 'rot_v']]
    
    min_event = curr_data.loc[curr_data.loc[:,'stim_speed']< 0, ['event', 'rot_v']]
    acc_rot_v = np.zeros((5000, 120)) 
    acc_rot_v.fill(np.nan)
    
    c=0
    for ev in np.unique(min_event.loc[:, 'event']):
        if ev == 0:
            continue
        curr_ev = curr_data.loc[curr_data.loc[:, 'event']==ev, 'rot_v']
        curr_ev = curr_ev.reset_index(drop=True)
        for i in range(0, len(curr_ev)-1):
            acc_rot_v[i,c] = sum(curr_ev[0:i])
            
        
        c=c+1
    
    plt.plot(acc_rot_v)
    plt.title(ID + 'four')
    plt.xlabel('frames')
    plt.ylabel('accumulated orientation change')
    plt.ylim([-6000, 6000])
    plt.show()
    # four_acc.loc[:,ID] = acc_rot_v
        
    # min_four_event = curr_data.loc[curr_data.loc[:,'stim_speed']== -4, ['event', 'rot_v']]
    # acc_rot_v_minfour = np.zeros((5000, 15)) 
    # acc_rot_v_minfour.fill(np.nan)

    # c=0
    # for ev in np.unique(min_four_event.loc[:, 'event']):
    #     curr_minfour_ev = curr_data.loc[curr_data.loc[:, 'event']==ev, 'rot_v']
    #     curr_minfour_ev = curr_minfour_ev.reset_index(drop=True)
    #     for i in range(0, len(curr_minfour_ev)-1):
    #         acc_rot_v_minfour[i,c] = sum(curr_minfour_ev[0:i])
            
        
    #     c=c+1
    
    # # minfour_acc.loc[:,ID]= acc_rot_v_minfour
    # plt.plot(acc_rot_v_minfour)
    # plt.title(ID + 'minus four')
    # plt.xlabel('frames')
    # plt.ylabel('accumulated orientation change')
    # plt.ylim([-6000, 6000])
    # plt.show()


    # prints ID of animal thats just finished so i can keep track while code is running
    print(ID)

    # get back to data folder so it can load next locust
    #os.chdir("Z:\DATA\experiment_trackball_Optomotor")
    os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data\Dots")

    # # Make trajectory plots
    # # BEWARE this is incredibly slow.
    # # gets all stimulus speeds and sorts them
    # speeds = pd.unique(curr_events.loc[:, 'speed'])
    # speeds = np.sort(speeds)

    # # gets color map for the trajectories
    # color = cm.get_cmap('viridis', 4500)

    # # loops through stimulus speeds and gets positions of  the events with that speed
    # for s in speeds:
    #   s=-4
    #   curr_speed= curr_events.loc[:, 'speed']== s
    #   curr_speed=np.where(curr_speed)

    # # makes a plot of 15 subplots for current stimulus speed
    #   fig, ax = plt.subplots(3,5, figsize=(40,20))
    #   fig.suptitle( 'Animal 01; speed:'+ str(s), fontsize=50)

    #  # row and column of subplot position
    #   row=0
    #   col=0

    # # loops through all events that have this speed
    #   for i in curr_speed[0]:
    #       # gets x and y for that event
    #       plotdatx= curr_data.loc[curr_data.loc[:,'event']==i+1, 'x_pos']
    #       plotdaty= curr_data.loc[curr_data.loc[:,'event']==i+1, 'y_pos']

    #       # loops through each row to give evry point a different color
    #       for p in range(0,len(curr_data.loc[curr_data.loc[:,'event']==i+1, 'x_pos'])-1):
    #           ax[row,col].scatter(plotdatx.iloc[p],plotdaty.iloc[p] , 1, color= color.colors[p])

    #    col = col + 1 # new column
    #    if col > 4: # when in every column of one row already is a plot go to next row
    #         col = 0
    #         row = row + 1

    # plt.show() #show plot


# acc_rot_v = acc_rot_v_minfour[1:3598, :]
# test= np.mean(acc_rot_v, axis=1)
# plt.plot(test)