import os, fnmatch
import pandas as pd
import numpy as np
def generate_curated_database(myList = [], *args):
    csv_pattern='trial_*.csv'
    dat_pattern='*.dat'
    overwrite_file=True
    track_ball_radius=4
    frame_rate=144
    for x in myList:
        # list up the files in the dir
        file = os.listdir(x)
        # look for files that matches the pattern
        csv_file= fnmatch.filter(file, csv_pattern)  
        dat_file= fnmatch.filter(file, dat_pattern)
        csv_file_directory= os.path.join(x ,csv_file[0])
        dat_file_directory= os.path.join(x ,dat_file[0])
        stim_info = pd.read_csv(csv_file_directory)
        stim_variable=[]
        stimulus_timestamp=[]
        for row in range(0, len(stim_info)):
        #delete the bracket in current speed and save it as integer in new list
            stim_variable.append(int(stim_info.iloc[row, 0].replace('(', '')))
        # combine the last two values into the timestamp (were divided into 2 columns cause german computer
        # used comma as decimal seperator and save timestamp as float in new file
            temp_str = str(stim_info.iloc[row, 1]) + '.' + stim_info.iloc[row, 2]
            stimulus_timestamp.append(float(temp_str.replace(')', '')))
        # 
        #stim_info.iloc[:, 0]= stim_variable
        #curr_events['speed']=curr_events['speed'].astype('int8')
        #stim_info.iloc[:, 1]= stimulus_timestamp
        
        
        ## load dat file
        ## fictrac title is here 'frame counter', 'delta rotation vector cam x', 'delta rotation vector cam y', \
        ##'delta rotation vector cam z', 'delta rotation error score', 'delta rotation vector lab x', \
        ##'delta rotation vector lab y', 'delta rotation vector lab z', 'absolute rotation vector cam x',\
        ##'absolute rotation vector cam y', 'absolute rotation vector cam z', \
        ##'absolute rotation vector lab x', 'absolute rotation vector lab y', \
        ##'absolute rotation vector lab z', 'intergrated x position','intergrated y position', \
        ##'intergrated animal heading', 'animal movement direction', 'animal movement speed', \
        ##'intergrated forward motion','intergrated side motion', 'timestamp', 'sequence counter','delta timestamp', 'alt. timestamp']

        ## load dat file
        raw_data = pd.read_table(dat_file_directory, sep="\s+")
        
        ## drop some column
        raw_data.drop(raw_data.columns[[0,1,2,3,4,8,9,10,16,17,18,19,20,21,22,23]], axis=1, inplace=True)
        raw_data.columns=['delta rotation vector lab x', 'delta rotation vector lab y', 'delta rotation vector lab z',                       'absolute rotation vector lab x', 'absolute rotation vector lab y', 'absolute rotation vector lab z',                       'intergrated x position','intergrated y position', 'timestamp']
        ## cleaning , and change the data from object to float
        raw_data.iloc[:, :]=raw_data.iloc[:, :].replace({',': ''}, regex= True)
        for ind in range(0, raw_data.shape[1]):
            raw_data.iloc[:, ind]=pd.to_numeric(raw_data.iloc[:, ind]).astype('float32')
        
        ## adjust the unit of the x, y position
        raw_data.iloc[:, 6]=raw_data.iloc[:, 6]*track_ball_radius
        raw_data.iloc[:, 7]=raw_data.iloc[:, 7]*track_ball_radius
        ## adjust the unit of the z vector based on the target frame rate of fictrac
        raw_data.iloc[:, 2]=raw_data.iloc[:, 2]*frame_rate

        # apply filter to the raw trace
        #curr_data.loc[:, ['x_pos']] = scipy.signal.savgol_filter(curr_data.loc[:, ['x_pos']], 99, 3, axis=0)
        #curr_data.loc[:, ['y_pos']] = scipy.signal.savgol_filter(curr_data.loc[:, ['y_pos']], 99, 3, axis=0)
        
        # compare the timestamp between event file and dat file
        data_timestamp=raw_data.loc[:, 'timestamp']
        stimulus_timestamp_b=np.tile(stimulus_timestamp,(len(data_timestamp),1))
        data_timestamp_b=np.tile(data_timestamp,(len(stimulus_timestamp),1))
        timestamp_diff= data_timestamp_b > np.transpose(stimulus_timestamp_b)
        stim_onset= np.where(np.roll(timestamp_diff,1,axis=1)!=timestamp_diff)[1][1::2]
        
        ### add new column here to store event info
        raw_data['stim_onset']=False
        raw_data.loc[stim_onset, 'stim_onset']=True
        raw_data['stim_variable']=0
        for ind in range(0, len(stimulus_timestamp)):
             raw_data.loc[stim_onset[ind], 'stim_variable']=stim_variable[ind]
        
        ### save the curated_database
        if overwrite_file==True:
            database_name='database_curated.pickle'
            database_directory= os.path.join(x,database_name)
            raw_data.to_pickle(database_directory)