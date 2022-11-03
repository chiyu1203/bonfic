# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:16:09 2022

@author: excid
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 08:27:32 2022

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

# Set directory to data folder

# os.chdir("Z:\DATA\experiment_trackball_Optomotor")
os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data\splitscreen")

calcnew = 1

# get all folders in data (all animal IDs)

animals = os.listdir()

# befaver = pd.DataFrame(np.zeros(shape=(len(animals)*8,3)), columns={'ID', 'befaft', 'average'})
# averages = pd.DataFrame(np.zeros(shape=(len(animals)*8,3)), columns={'ID', 'dot_num', 'average'})
# c=0
# a=0
print(animals) # to see which animals

# loop through animals
for ID in animals:
    
    # go into animal folder
    os.chdir(ID)
    
    
    # if not ID=='L22056' and not ID=='L22057' and not ID=='L22058' :
    #         print('skip ' + ID )
    #         os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data/densities")
    #         continue
    #get all days in animals folder
    

    if  not os.listdir('data') or calcnew == 1:
        # loop through days (currently only one per animal, but might change)
        
        #go into day folder
        
    
        # if os.listdir('data'):
        #     print('skip ' + ID + ' ' + day)
        #     os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data")
        #     continue
    
    
        # load trial event
        tr_ev = glob.glob('*.{}'.format('csv'))
        curr_events_raw = pd.read_csv(tr_ev[0])
        
        curr_events_raw=curr_events_raw.iloc[::2,:]
        curr_events_raw=curr_events_raw.reset_index()
    
        # make a dataframe of zeros
        zeros = np.zeros(shape=(len(curr_events_raw), 10))
        curr_events = pd.DataFrame(zeros, columns={'timestamp','bar', 'speed',  'dot_num_l', 'dot_num_r', 'disttr', 'rot_v', 'rot_v_bar', 'stimulus', 'avg_speed'})
    
        # loop through rows of event trials and save speed and timestamp to new dataframe
        for row in range(0, len(curr_events_raw)):
            #delete the bracket in current speed and save it as integer in new file
            curr_events.loc[:, 'speed'].iloc[row] = int(curr_events_raw.iloc[row,14].split("=",1)[1])
            curr_events.loc[:, 'dot_num_l'].iloc[row] = int(curr_events_raw.iloc[row,8].split("=",1)[1])
            curr_events.loc[:, 'dot_num_r'].iloc[row] = int(curr_events_raw.iloc[row,7].split("=",1)[1])
            
            
            # combine the last two values into the timestamp (were divided into 2 columns cause german computer
            # used comma as decimal seperator
            if isinstance(curr_events_raw.loc[row, 'Value'], float):
                timestamp = curr_events_raw.loc[row, 'Timestamp']
            else:
                timestamp = str(curr_events_raw.loc[row, 'Timestamp']) + '.' + curr_events_raw.loc[row, 'Value']
    
            # save timestamp as float in new file
            curr_events.loc[:, 'timestamp'].iloc[row] = float(timestamp.replace(')', ''))
        
        conditions = [(curr_events.loc[:,'dot_num_l']==256) & (curr_events.loc[:,'dot_num_r']==256),
                      (curr_events.loc[:,'dot_num_l']==256) & (curr_events.loc[:,'dot_num_r'] ==64),
                      (curr_events.loc[:,'dot_num_l']==256) & (curr_events.loc[:,'dot_num_r']==4096),
                      (curr_events.loc[:,'dot_num_l']==64) & (curr_events.loc[:,'dot_num_r'] == 256),
                      (curr_events.loc[:,'dot_num_l'] == 4096) & (curr_events.loc[:,'dot_num_r'] == 256)]
        
        choices = ['equal', 'right low', 'right high', 'left low', 'left high']
        curr_events.loc[:,'stimulus']=np.select(conditions, choices)
        
        curr_events.loc[:,'bar']= 'bar'
        # get dat file
        dat = glob.glob('*.{}'.format('dat'))
        curr_data_raw = pd.read_table(dat[0], sep="\s+")
    
        # create dataframe of zeros
        zeros = np.zeros(shape=(len(curr_data_raw), 11))
        curr_data = pd.DataFrame(zeros, columns={'timestamp', 'x_pos', 'y_pos', 'event', 'stim_speed', 'rot_v', 'dist', 'head', 'rot_err', 'speed', 'stimulus'})
    
        # assign values to columns
        curr_data.loc[:, 'timestamp'] = curr_data_raw.iloc[:,24] # timestamp later used to match data to event
    
        curr_data.loc[:, 'x_pos'] = curr_data_raw.iloc[:, 14].replace({',': ''}, regex= True) # x position from fictrac data has a comma at end. has to be deleted
        curr_data.loc[:, 'y_pos'] = curr_data_raw.iloc[:, 15].replace({',': ''}, regex= True) # y position from fictrac data as a comma at end. has to be deleted
        curr_data.loc[:, 'x_pos'] = pd.to_numeric(curr_data.loc[:, 'x_pos']) * 4 # positions should be numeric (wasn't previously due to comma idk why it was there)
        curr_data.loc[:, 'y_pos'] = pd.to_numeric(curr_data.loc[:, 'y_pos']) * 4    # four is radius of trackball
        curr_data.loc[:, 'rot_err'] = curr_data_raw.iloc[:, 4].replace({',': ''}, regex= True)
        curr_data.loc[:, 'rot_err'] = pd.to_numeric(curr_data.loc[:, 'rot_err'])
        # curr_data.loc[:, 'speed'] = curr_data_raw.iloc[:, 18].replace({',': ''}, regex= True)
        # curr_data.loc[:, 'speed'] = pd.to_numeric(curr_data.loc[:, 'speed'])
        
        
        
        curr_data.loc[:, 'rot_v'] = curr_data_raw.iloc[:, 7].replace({',': ''}, regex=True) # rotation around z axis
        curr_data.loc[:, 'rot_v'] = pd.to_numeric(curr_data.loc[:, 'rot_v']) * 144
        
        curr_data.loc[:, 'head'] = curr_data_raw.iloc[:, 16].replace({',': ''}, regex=True) # heading
        curr_data.loc[:, 'head'] = pd.to_numeric(curr_data.loc[:, 'head']) * 144
        
        #apply filter to the raw trace
        curr_data.loc[:, ['x_pos']] = scipy.signal.savgol_filter(curr_data.loc[:, ['x_pos']], 19, 3, axis=0)
        curr_data.loc[:, ['y_pos']] = scipy.signal.savgol_filter(curr_data.loc[:, ['y_pos']], 19, 3, axis=0)
        #curr_data.loc[:, ['rot_v']] = scipy.signal.savgol_filter(curr_data.loc[:, ['rot_v']], 99, 3, axis=0)
        
        # curr_data.loc[:, 'dist'] = curr_data_raw.iloc[:, 18].replace({',': ''}, regex= True)
        # curr_data.loc[:, 'dist'] = pd.to_numeric(curr_data.loc[:, 'dist'])*4
        # plt.subplot(222)
        # plt.plot(curr_data.loc[:, ['x_pos']], curr_data.loc[:, ['y_pos']])
        # # plt.plot(x_pos_19, y_pos_19)
        # # plt.subplot(223)
        # # plt.plot(x_pos_59, y_pos_59)
        # # plt.subplot(224)
        # # plt.plot(x_pos_99, y_pos_99)
        # plt.show()
    
        #calculate distance between consecutive points
        diff_dat = curr_data.loc[:, ['x_pos', 'y_pos']].diff()
        curr_data.loc[:, 'dist']= np.sqrt(diff_dat.loc[:,'x_pos'].pow(2) + diff_dat.loc[:,'y_pos'].pow(2))
        
        #speeds
        curr_data.loc[:,'speed'] = curr_data.loc[:,'dist'] *144
        
        
        
        # heading change
        
        curr_data.loc[:, 'headc'] = curr_data.loc[:, 'head'].diff()
        
        curr_data.loc[curr_data.loc[:, 'headc']> 2.967, 'headc'] = np.nan
        curr_data.loc[curr_data.loc[:, 'headc']< -2.967, 'headc'] = np.nan
        
        # gets the differences between timesteps of events
        
        differences = curr_events.loc[:, 'timestamp'].diff()
    
        # the events are all supposed to be 2minutes apart, so its divided by 2 to get the 1min step
        five_steps = differences / 2
    
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
            curr_data.loc[(curr_data.loc[:, 'timestamp'] > event) & (curr_data.loc[:, 'timestamp'] < (next_event-five_step)), 'stimulus'] = curr_events.loc[ev_num, 'stimulus']
            # ev dat is the x, y pos. and rotational velocity of current event when stimulus is present
            ev_dat = curr_data.loc[(curr_data.loc[:, 'event'] == (ev_num+1)) & (curr_data.loc[:, 'stim_speed'] != 0), ['x_pos', 'y_pos', 'rot_v', 'dist', 'speed']]
            bar_dat = curr_data.loc[(curr_data.loc[:, 'event'] == (ev_num+1)) & (curr_data.loc[:, 'stim_speed'] == 0), ['x_pos', 'y_pos', 'rot_v', 'dist', 'speed']]
            # calculates disttr within that event
            curr_events.loc[ev_num, 'disttr'] = np.nansum(ev_dat.loc[:, 'dist'])
            
            curr_events.loc[ev_num, 'avg_speed'] = np.nanmean(ev_dat.loc[:, 'speed'])
    
            ev_dat = ev_dat.iloc[0:3000, :]
            bar_dat = bar_dat.iloc[0:3000, :]
    
            # calculates optomotor response index
            curr_events.loc[ev_num, 'rot_v'] = (np.nansum(abs(ev_dat.loc[ev_dat.loc[:,'rot_v']<0,'rot_v'])) - np.nansum(abs(ev_dat.loc[ev_dat.loc[:,'rot_v']>0,'rot_v'])))/(np.nansum(abs(ev_dat.loc[ev_dat.loc[:,'rot_v']<0,'rot_v'])) + np.nansum(abs(ev_dat.loc[ev_dat.loc[:,'rot_v']>0,'rot_v'])))
            curr_events.loc[ev_num, 'rot_v_bar'] = (np.nansum(abs(bar_dat.loc[bar_dat.loc[:,'rot_v']<0,'rot_v'])) - np.nansum(abs(bar_dat.loc[bar_dat.loc[:,'rot_v']>0,'rot_v'])))/(np.nansum(abs(bar_dat.loc[bar_dat.loc[:,'rot_v']<0,'rot_v'])) + np.nansum(abs(bar_dat.loc[bar_dat.loc[:,'rot_v']>0,'rot_v'])))
        # # Plots to figure out threshold for events ( decided on 1)
        # plt.hist(curr_events.loc[:, 'disttr'], bins = 20)
        # plt.xlabel('distance travelled')
        # plt.show()
        
        
        # saves data so this doesnt have to be run for every future plot
        os.chdir('data')
        # curr_data.to_csv('data.csv')
        # curr_events.to_csv('events.csv')
        
        
        
    else:
        os.chdir('data')
        print('load data')
        
        curr_events = pd.read_csv('events.csv')
        curr_data = pd.read_csv('data.csv')

    print('percent of events under treshold')
    print(((len(curr_events.loc[curr_events.loc[:, 'disttr'] <= 15, ['speed', 'disttr']]))/18)*100)
    curr_events = curr_events.loc[curr_events.loc[:, 'disttr'] > 15, :]
    
    # curr_events= curr_events.reset_index(drop=True)
    # curr_events = curr_events.loc[curr_events.loc[:, 'disttr'] > 5, :]
    # curr_events.loc[curr_events.loc[:,'speed']<0,'dot_num'] =  curr_events.loc[curr_events.loc[:,'speed']<0,'dot_num'] * -1
    
    # curr_events.loc[:,'ev_before']= curr_events.loc[:, 'speed'].shift()
    # conditions = [(curr_events.loc[:,'speed']<0) & (curr_events.loc[:,'ev_before']<0),
    # (curr_events.loc[:,'speed']<0) & (curr_events.loc[:,'ev_before']>0),
    # (curr_events.loc[:,'speed']>0) & (curr_events.loc[:,'ev_before']>0),
    # (curr_events.loc[:,'speed']>0) & (curr_events.loc[:,'ev_before']<0)]
        
    # choices = ['negneg', 'posneg', 'pospos', 'negpos']
    
    # curr_events.loc[:, 'befaft']=np.select(conditions, choices, 'NA').T
    
    # if not len(np.unique(curr_events.loc[:,'dot_num']))== 8:
    #     print(ID + ': not all events')
    #     os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data/densities")
    #     continue
    # ax =sns.stripplot(x="befaft", y="rot_v", data=curr_events.loc[:, ['befaft', 'rot_v']], dodge=True, jitter=True, alpha=.8, zorder=1, size=4)
    # ax =sns.pointplot(x='befaft', y='rot_v', ci='sd', data=curr_events.loc[:, ['befaft', 'rot_v']], join=False, scale=0.5, zorder=100, color='black', errwidth=1)
    # plt.show()
    
    # for num in np.unique(curr_events.loc[:,'befaft']):
    #     befaver.loc[a, 'ID']= ID
    #     befaver.loc[a, 'befaft']= num
    #     befaver.loc[a, 'average']= curr_events.loc[curr_events.loc[:,'befaft']==num, 'rot_v'].mean()
    #     a=a+1
    
    # dotdens= [-4096,-1024, -256, -64, 64, 256, 1024, 4096]    
    # for num in dotdens:
    #     averages.loc[c, 'ID']= ID
    #     averages.loc[c, 'dot_num']= num
    #     averages.loc[c, 'average']= curr_events.loc[curr_events.loc[:,'dot_num']==num, 'rot_v'].mean()
    #     c=c+1
    
    
    
    # # # Speed histogram
    # n, bins, patches = plt.hist(curr_data.loc[:, 'speed'], 20, range =(0, 2), density=True, facecolor='b', edgecolor= 'b',alpha=0.25, width= 0.05)
    # # plt.ylim((0,8))
    # plt.xlabel('speeds')
    # plt.title(ID + ' - speed distribution across all trials')
    # plt.savefig('speedhist.svg')
    # plt.show()
        
        
    # # Dist histogram
    # plt.hist(curr_events.loc[:, 'disttr'], bins = 20)
    # plt.xlabel(ID + ' - distance travelled')
    # plt.savefig('disthist.svg')
    # plt.show()
 
    
    # if len(np.unique(curr_events.loc[:,'dot_num1'])) != 3:
    #     print('skip ' + ID + ': not all events') 
    #     os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data/densities")
    #     continue
       
    # if not len(np.unique(curr_events.loc[:,'dot_num1']))== 3:
    #     print(ID + ': not all events')
    #     os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data/densities")
    #     continue

    bar_events = curr_events.copy()
    bar_events.stimulus = 'bar'
    bar_events.rot_v = bar_events.rot_v_bar
    curr_events = pd.concat([curr_events,bar_events], axis=0)
    curr_events.stimulus = curr_events.stimulus.astype('category')
    sorter= ['right high', 'left low', 'equal', 'bar', 'right low', 'left high']
    curr_events.stimulus = curr_events.stimulus.cat.set_categories(sorter)
    curr_events = curr_events.sort_values(by=['stimulus'])
    ax =sns.stripplot(x="stimulus", y="rot_v", data=curr_events.loc[:, ['stimulus', 'rot_v']], dodge=True, jitter=True, alpha=.8, zorder=1, size=4)
    ax =sns.pointplot(x='stimulus', y='rot_v', ci='sd', data=curr_events.loc[:, ['stimulus', 'rot_v']], join=False, scale=0.5, zorder=100, color='black', errwidth=1)
    
   
    
    plt.ylim([-1, 1])
    #ax.set_xticklabels([ '4096', '1024', '256','64', '64', '256', '1024', '4096'])
    # plt.xticks()
    #plt.savefig('optomotor.svg')
    plt.show()
    
    # min_event = curr_data.loc[curr_data.loc[:,'stimulus']== 'equal', ['event', 'rot_v']]
    # acc_rot_v = np.zeros((50000, 120)) 
    # acc_rot_v.fill(np.nan)
   
    # c=0
    # for ev in np.unique(min_event.loc[:, 'event']):
    #     if ev == 0:
    #         continue
    #     curr_ev = curr_data.loc[curr_data.loc[:, 'event']==ev, 'rot_v']
    #     curr_ev = curr_ev.reset_index(drop=True)
    #     for i in range(0, len(curr_ev)-1):
    #         acc_rot_v[i,c] = np.nansum(curr_ev[0:i])
            
    
    #     c=c+1
    
    # plt.plot(acc_rot_v)
    # plt.title(ID + ' - accumulated response ')
    # plt.xlabel('frames')
    # plt.ylabel('accumulated orientation change')
    # plt.ylim([-6000, 6000])
    # #plt.savefig('accormin.svg')
    # plt.show()
    
        
    # min_four_event = curr_data.loc[curr_data.loc[:,'stim_speed']>0, ['event', 'rot_v']]
    # acc_rot_v_minfour = np.zeros((5500, 120)) 
    # acc_rot_v_minfour.fill(np.nan)

    # c=0
    # for ev in np.unique(min_four_event.loc[:, 'event']):
    #     if ev == 0:
    #         continue
    #     curr_minfour_ev = curr_data.loc[curr_data.loc[:, 'event']==ev, 'rot_v']
    #     curr_minfour_ev = curr_minfour_ev.reset_index(drop=True)
    #     for i in range(0, len(curr_minfour_ev)-1):
    #         acc_rot_v_minfour[i,c] = np.nansum(curr_minfour_ev[0:i])
            
        
    #     c=c+1
    
    # # minfour_acc.loc[:,ID]= acc_rot_v_minfour
    # plt.plot(acc_rot_v_minfour)
    # plt.title(ID + ' - accumulated response for dots moving right')
    # plt.xlabel('frames')
    # plt.ylabel('accumulated orientation change')
    # plt.ylim([-6000, 6000])
    # plt.savefig('accorplus.svg')
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
    
    


    # prints ID of animal thats just finished so i can keep track while code is running
    print(ID)

    # get back to data folder so it can load next locust
    #os.chdir("Z:\DATA\experiment_trackball_Optomotor")
    os.chdir("C:/Users/excid/Documents/University/Master/experiment/trackball/data/splitscreen")

    # Make trajectory plots
    # BEWARE this is incredibly slow.
    # gets all stimulus speeds and sorts them
    # speeds = pd.unique(curr_events.loc[:, 'dot_num'])
    # speeds = np.sort(speeds)

    # # gets color map for the trajectories
    # color = cm.get_cmap('viridis', 4500)

    # # loops through stimulus speeds and gets positions of  the events with that speed
    # for s in speeds:
      
    #   curr_speed= curr_events.loc[:, 'dot_num']== s
    #   curr_speed=np.where(curr_speed)

    # # makes a plot of 15 subplots for current stimulus speed
    #   fig, ax = plt.subplots(3,5, figsize=(40,20))
    #   fig.suptitle( 'Animal 01; dot_num:'+ str(s), fontsize=50)

    #   # row and column of subplot position
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

    #       col = col + 1 # new column
    #       if col > 4: # when in every column of one row already is a plot go to next row
    #           col = 0
    #           row = row + 1

    # plt.show() #show plot


# acc_rot_v = acc_rot_v_minfour[1:3598, :]
# test= np.mean(acc_rot_v, axis=1)
# plt.plot(test)