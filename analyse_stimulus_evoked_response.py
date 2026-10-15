"""spontaneous response: needs to figure out stationary, moving, timing and total travel distance as a overview 
event evoked response: needs to divide stationary trails, moving trails 
(in the case of optomotor response, need to subdivide them into moving along stimulus, 
moving against stimulus and others)
event evoked response: need to output the trial id (both within the same stimulus type and overall) and response, 
velocity vector across time in a trial plus optomotor response
analyis methods needs to tell us what to analye. For example optomotor response, acceleration epochs, deceleration epochs, transition between stationary to walk and walk to stationary
"""
import matplotlib
matplotlib.use("Agg")
import json, sys, re, time, warnings, gzip
import pandas as pd
import numpy as np
from scipy.io import savemat
from scipy.signal import savgol_filter,medfilt
from scipy.stats import circmean
import scipy.stats as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.transforms as mt
from pathlib import Path
from align_data_with_ttl import analyse_led_signal

##need to add this additional cell because useful tools are in another folder. Need to integrate these two folders one day
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
if sys.platform == "darwin":
    utilities_string = "/utilities"
else:
    utilities_string = "\\utilities"
sys.path.insert(0, str(parent_dir) + utilities_string)
from useful_tools import find_file, find_nearest, calculate_fft
from data_cleaning import sorting_trial_info, load_temperature_data,euclidean_distance,diskretize,bfill

class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def plot_yaw_angular_velocity(df,analysis_methods,init_frame=1000,end_frame=3500,vr_no="all",colormap_name="coolwarm"):
    save_output= analysis_methods.get("save_output")
    visual_paradigm_name= analysis_methods.get("experiment_name")
    camera_fps=analysis_methods.get("camera_fps")
    active_trials_only=analysis_methods.get("active_trials_only")
    COL = MplColorHelper(colormap_name, 0, len(df["stim_type"].unique())-1)
    sm = cm.ScalarMappable(cmap=colormap_name)
    analysis_window=analysis_methods.get("analysis_window")
    stim_duration=analysis_methods.get("stim_duration")
    using_bs_ci=True
    use_mean=False
    stimulus_velocity=64 #unit degree/s
    walk_threshold_tbt=10
    premature_turn=0.25#set this criteria to remove animals starting to make turn turing ISI
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 10))
    ax1, ax2 = axes.flatten()

    fig_tbt, subplots = plt.subplots(
        nrows=1, ncols=len(df["stim_type"].unique())+1, figsize=(20, 4), tight_layout=True
    )
    if active_trials_only==True:
        #df_selected=df[(df['travel_distance_isi']>walk_threshold_tbt) & (abs(df["opto_index_isi"])<premature_turn)]
        df_selected=df[(df['travel_distance']>walk_threshold_tbt)]
    else:
        df_selected=df
    i=0
    for this_stim, entries in df_selected.groupby('stim_type'):
        these_angular_velocity=entries.loc[:,'angular_velocity_f0':].values
        these_angular_acceleration=np.gradient(these_angular_velocity,1/camera_fps,axis=1)
        these_headings=np.cumsum(these_angular_velocity,axis=1)
        time_points = np.arange(these_headings.shape[1])
        time_points_2D = np.tile(time_points, (these_headings.shape[0], 1))
        #define the stimulus speed in the first iteration
        if i ==0:
            xticks = [0, init_frame, end_frame]
            xticklabels = [str(analysis_window[0]), "0", str(stim_duration)]
            y_range=150
            tmp = np.full(these_headings.shape[1],np.nan)
            dif_tmp= np.full(these_angular_velocity.shape[1],np.nan)
            stim_w_array=np.ones(end_frame-init_frame)*stimulus_velocity*np.pi/180
            tmp[init_frame:end_frame]=np.cumsum(stim_w_array)
            dif_tmp[init_frame:end_frame]=stim_w_array


        #plot trial-by-trial response
        #ax.scatter(time_points_2D, these_headings, s=0.01,alpha=0.1,color=COL.get_rgb(i))
        #plot the whole area (not recommended for tbt)
        #ax.plot(time_points_2D, these_headings, linestyle='-',linewidth=0.1,color=COL.get_rgb(i)))
        subplots[i].scatter(time_points_2D, these_headings, s=0.01,alpha=0.1,color=COL.get_rgb(i))
        subplots[i].plot(time_points,tmp,color='k',linewidth=0.8)
        subplots[i].plot(time_points,tmp*-1,color='k',linewidth=0.8)
        subplots[i].axvline(
        x=init_frame,
        color="black",
        linestyle="--",
        label="Stimulus Starts",
        )
        subplots[i].axvline(
            x=end_frame,
            color="black",
            linestyle="--",
            label="Stimulus Ends",
        )
        subplots[i].set(
        ylabel="Orientation (rad)",
        xticks=xticks,
        xticklabels=xticklabels,
        xlabel="Time (s)",
    )
        #plot average response
        if use_mean==True:
            headings_to_plot = np.mean(these_headings, axis=0)
            velocity_to_plot = np.mean(these_angular_velocity, axis=0)
            acceleration_to_plot=np.mean(these_angular_acceleration, axis=0)
        else:
            headings_to_plot=np.median(these_headings, axis=0)
            velocity_to_plot=np.median(these_angular_velocity, axis=0)
            acceleration_to_plot=np.median(these_angular_acceleration, axis=0)
        #ax.plot(time_points,headings_to_plot,color=COL.get_rgb(i))
        ax1.plot(time_points,velocity_to_plot,color=COL.get_rgb(i))
        ax2.plot(time_points,acceleration_to_plot,color=COL.get_rgb(i))
        subplots[len(df["stim_type"].unique())].plot(time_points,headings_to_plot,color=COL.get_rgb(i))

        ##to plot distribution with bootstrap 95% confidence interval
        if using_bs_ci:
            # data=(these_headings,)
            # res=st.bootstrap(data,np.std,n_resamples=99,vectorized=True,random_state=1)
            # y1=res.confidence_interval[0]
            # y2=res.confidence_interval[1]
            # del data
            confidence_level = 0.95
            #data=(these_angular_velocity,)
            res=st.bootstrap(data=(these_angular_velocity,),statistic=np.median,n_resamples=999,vectorized=True,batch=100,random_state=1)
            w_y1=res.confidence_interval[0]
            w_y2=res.confidence_interval[1]
            ##to plot distribution with 95% confidence interval with normal distribution because it is faster not because it is correct
            cl95=st.norm.interval(confidence_level,loc=headings_to_plot,scale=st.sem(these_headings))
            y1=cl95[0][:]
            y2=cl95[1][:]
            #cl95=st.norm.interval(confidence_level,loc=velocity_to_plot,scale=st.sem(these_angular_velocity))
            # w_y1=cl95[0][:]
            # w_y2=cl95[1][:]
            cl95=st.norm.interval(confidence_level,loc=acceleration_to_plot,scale=st.sem(these_angular_acceleration))
            acc_y1=cl95[0][:]
            acc_y2=cl95[1][:]
        else:
            ##to plot distribution with sem
            sem_response = np.std(these_headings, axis=0, ddof=1) / np.sqrt(these_headings.shape[0])
            y1 = headings_to_plot + sem_response
            y2 = headings_to_plot - sem_response
            sem_response = np.std(these_angular_velocity, axis=0, ddof=1) / np.sqrt(these_angular_velocity.shape[0])
            w_y1=velocity_to_plot + sem_response
            w_y2=velocity_to_plot - sem_response
            sem_response = np.std(these_angular_acceleration, axis=0, ddof=1) / np.sqrt(these_angular_acceleration.shape[0])
            acc_y1=acceleration_to_plot + sem_response
            acc_y2=acceleration_to_plot - sem_response
        #ax.fill_between(time_points,y1,y2, alpha=0.5,color=COL.get_rgb(i))
        ax1.fill_between(time_points,w_y1,w_y2, alpha=0.5,color=COL.get_rgb(i))
        ax2.fill_between(time_points,acc_y1,acc_y2, alpha=0.5,color=COL.get_rgb(i))
        subplots[len(df["stim_type"].unique())].fill_between(time_points,y1,y2, alpha=0.5,color=COL.get_rgb(i))

        i=i+1
    # ax.plot(time_points,tmp,color='k',linewidth=0.8)
    # ax.plot(time_points,tmp*-1,color='k',linewidth=0.8)
    ax1.plot(time_points,dif_tmp,color='k',linewidth=0.8)
    ax1.plot(time_points,dif_tmp*-1,color='k',linewidth=0.8)
    subplots[len(df["stim_type"].unique())].plot(time_points,tmp,color='k',linewidth=0.8)
    subplots[len(df["stim_type"].unique())].plot(time_points,tmp*-1,color='k',linewidth=0.8)
    # ax.axvline(
    #     x=init_frame,
    #     color="black",
    #     linestyle="--",
    #     label="Stimulus Starts",
    # )
    # ax.axvline(
    #     x=end_frame,
    #     color="black",
    #     linestyle="--",
    #     label="Stimulus Ends",
    # )
    ax1.axvline(
        x=init_frame,
        color="black",
        linestyle="--",
        label="Stimulus Starts",
    )
    ax1.axvline(
        x=end_frame,
        color="black",
        linestyle="--",
        label="Stimulus Ends",
    )
    ax2.axvline(
        x=init_frame,
        color="black",
        linestyle="--",
        label="Stimulus Starts",
    )
    ax2.axvline(
        x=end_frame,
        color="black",
        linestyle="--",
        label="Stimulus Ends",
    )
    subplots[len(df["stim_type"].unique())].axvline(
        x=init_frame,
        color="black",
        linestyle="--",
        label="Stimulus Starts",
    )
    subplots[len(df["stim_type"].unique())].axvline(
        x=end_frame,
        color="black",
        linestyle="--",
        label="Stimulus Ends",
    )
    # ax.set(
    #     ylabel="Orientation (rad)",
    #     xticks=xticks,
    #     xticklabels=xticklabels,
    #     xlabel="Time (s)",
    # )
    ax1.set(
        ylabel=r'angular velocity (rad s$^{-1}$)',
        xticks=xticks,
        yticks=[-1,0,1],
        #ylim=[-0.005,0.02],
        ylim=[-1.2,1.2],
        xticklabels=xticklabels,
        xlabel="Time (s)",
    )
    ax2.set(
        ylabel=r'angular acceleration (rad s$^{-2}$)',
        xlabel="Time (s)",
    )
    subplots[len(df["stim_type"].unique())].set(
        xticks=xticks,
        #yticks=[-2000,0,2000],
        xticklabels=xticklabels,
        xlabel="Time (s)")
    tick_positions = np.linspace(0, 1, len(df["stim_type"].unique()))
    these_coherence=df["stim_type"].unique()
    these_coherence.sort()
    tick_labels = [str(value) for value in these_coherence]
    cbar = plt.colorbar(sm, ax=[ax1,ax2], orientation="horizontal", location='top',ticks=tick_positions,aspect=80)
    #cbar = plt.colorbar(sm, ax=[ax2], orientation="horizontal", location='top',ticks=tick_positions,aspect=60)
    cbar.set_ticklabels(tick_labels)
    #cbar.ax.tick_params(labelsize=14)
    if use_mean:
        fig_name=f"{vr_no}_bonfic_{visual_paradigm_name}_mean_dynamics.png"
        fig_tbt_name=f"{vr_no}_bonfic_{visual_paradigm_name}_mean_heading.png"
    else:
        fig_name=f"{vr_no}_bonfic_{visual_paradigm_name}_median_dynamics.png"
        fig_tbt_name=f"{vr_no}_bonfic_{visual_paradigm_name}_median_heading.png"
    if save_output==True:
        fig.savefig(fig_name)
        fig_tbt.savefig(fig_tbt_name)

def a_guest(x):
    # Initialize an array of zeros with the same shape as x
    b = np.zeros_like(x, dtype=bool)

    # Find the index of the maximum value in each row of x
    i = np.argmax(x, axis=1)

    # Set the value at the index of the maximum value to True for each row of x
    b[np.arange(i.size), i] = np.logical_or.reduce(x, axis=1)

    return b


def analysis_stim_evoked_response_combo(
    stim_to_analyse,
    stimulus_info,
    yaw_angular_velocity,
    travel_distance_fbf,
    tem_df_this_exp,
    analysis_methods

):
    analysis_window=analysis_methods.get("analysis_window")
    camera_fps=analysis_methods.get("camera_fps")
    init_frame = abs(analysis_window[0]) * camera_fps
    end_frame = analysis_window[1] * camera_fps + init_frame
    duration_for_optomotor_index=analysis_methods.get("duration_for_optomotor_index")
    angular_velocity_tbt = extract_stim_locked_data(
        stim_to_analyse, yaw_angular_velocity, analysis_window, camera_fps
    )
    travel_distance_tbt= extract_stim_locked_data(
        stim_to_analyse,
        travel_distance_fbf,
        analysis_window,
        camera_fps,
    )
    if analysis_methods.get("temperature_data_options") == "trial":
        if analysis_methods.get("experiment_name").lower() !="coherence":
            temp_tbt=stimulus_info['temperature'].values
            humid_tbt=stimulus_info['humidity'].values
        else:
            temp_tbt=stimulus_info['temperature'][1::2].values
            humid_tbt=stimulus_info['humidity'][1::2].values
    elif len(tem_df_this_exp) > 0:
        tmp1= extract_stim_locked_data(
            stim_to_analyse,
            tem_df_this_exp[:, 0],
            analysis_window,
            camera_fps,
        )
        tmp2= extract_stim_locked_data(
            stim_to_analyse,
            tem_df_this_exp[:, 1],
            analysis_window,
            camera_fps,
        )
        temp_tbt=np.nanmean(tmp1, axis=1)
        humid_tbt=np.nanmean(tmp2, axis=1)
    else:
        temp_tbt = np.empty((stimulus_info.shape[0]))
        temp_tbt[:] = np.nan
        humid_tbt = np.empty((stimulus_info.shape[0]))
        humid_tbt[:] = np.nan

    #######Note: opto_index_isi and travel_distance_isi is defined based either the first element of analysis window or 5 sec, depends on which one is shorter.
    shortened_isi=5
    if shortened_isi>abs(analysis_window[0]):
        shortened_isi=abs(analysis_window[0])
    summary_list={
        "stim_index": stim_to_analyse.index,
        "temperature": temp_tbt,
        "humidity":humid_tbt
    }
    if duration_for_optomotor_index>0:
        opto_all_trials, opto_all_isi = probes_optomotor_index(
        angular_velocity_tbt, init_frame,end_frame,camera_fps, shortened_isi
    )
        opto_dict={"opto_index_isi": opto_all_isi,
            "opto_index": opto_all_trials}
        summary_list.update(opto_dict) 
    return (
        summary_list,
        angular_velocity_tbt,
        travel_distance_tbt,
        init_frame,
        end_frame,
    )
def define_analysis_window(stim_duration,interval_duration):
    if type(interval_duration) == list and type(stim_duration) == list:
                analysis_window = [
                    min(interval_duration) * -1,
                    max(stim_duration) + min(interval_duration),
                ]
    elif type(interval_duration) == list:
                analysis_window = [
                    min(interval_duration) * -1,
                    stim_duration + min(interval_duration),
                ]
    elif type(stim_duration) == list:
                analysis_window = [
                    interval_duration * -1,
                    max(stim_duration) + interval_duration,
                ]
    else:
                analysis_window = [
                    interval_duration * -1,
                    stim_duration + interval_duration,
                ]
    return analysis_window


def detect_stop_events(arr, analysis_methods,response_window=[-0.5,4],sustained_time_for_behaviour=0.5,save_path=Path.cwd()):
    analysis_window=analysis_methods.get("analysis_window")
    camera_fps=analysis_methods.get("camera_fps")
    plotting_tbt_overview=analysis_methods.get("plotting_tbt_overview")
    n_trials = arr.shape[0]
    event_detected = np.zeros(n_trials, dtype=bool)
    event_index = np.full(n_trials, -1, dtype=int)
    event_time = np.full(n_trials, np.nan)
    # Define frame number for analysis windows
    relative_time=np.arange(start=analysis_window[0],stop=analysis_window[1],step=(analysis_window[1]-analysis_window[0])/arr.shape[1])
    sustain_samples = int(sustained_time_for_behaviour*camera_fps)
    analysis_mask = (relative_time > response_window[0]) & (relative_time <= response_window[1])
    analysis_indices = np.where(analysis_mask)[0]
    if arr.dtype == 'float64':## feed raw velocity data. Otherwises, use boolean value of walk/stop states
        varying_deceleration_threshold=False
        if varying_deceleration_threshold:
            baseline_mask = (relative_time >= -1.5) & (relative_time < response_window[0])
            baseline_data = arr[:, baseline_mask]
            baseline_mean = baseline_data.mean(axis=1)
            baseline_std2 = 2 * baseline_data.std(axis=1, ddof=1)
            threshold = baseline_mean - baseline_std2
        else:
            threshold = 0.5*np.ones(n_trials)
        # ------------------------------------------------------------------
        # Detect sustained threshold crossing
        # ------------------------------------------------------------------
        for trial in range(n_trials):

            th = threshold[trial]
            trace = arr[trial]
            if plotting_tbt_overview:
                plt.figure(figsize=(8, 6))
                plt.plot(np.arange(trace.shape[0]),trace, linewidth=0.2)
                plt.savefig(save_path / f'velocity_trial{trial}.png')
                plt.close()
            for idx in analysis_indices:

                # Not enough samples left for confirmation
                # if idx + sustain_samples >= len(trace):
                #     break

                if trace[idx] < th:

                    # Require all following samples (0.5 sec) remain below threshold
                    if np.all(trace[idx:idx + sustain_samples + 1] < th):
                        event_detected[trial] = True
                        event_index[trial] = idx
                        event_time[trial] = relative_time[idx]
                        if plotting_tbt_overview:
                            plt.figure(figsize=(8, 6))
                            plt.plot(np.arange(trace.shape[0]),trace, linewidth=0.2)
                            plt.plot(idx,trace[idx],'o',color='r')
                            plt.savefig(save_path / f'velocity_trial{trial}.png')
                            plt.close()
                        break
    else:
        walk_state=arr[:,analysis_indices]
        (id,frame_number)=np.where(np.diff(walk_state))
        event_index[id]=frame_number
        if plotting_tbt_overview:
            for trial in range(n_trials):
                trace = arr[trial]
                plt.figure(figsize=(8, 6))
                plt.plot(np.arange(trace.shape[0]),trace, linewidth=0.2)
                if trial in id:
                    plt.plot(event_index[trial]+int(camera_fps*(abs(analysis_window[0])+response_window[0])),0.5,'o',color='r')
                plt.savefig(save_path / f'walk_state_trial{trial}.png')
                plt.close()
    return event_index

def recovery_detection(dev_trial, dev_timing_from_analysis_window):
    velocity_threshold = 5
    ##apply these velocity traces with velocity threshold
    stationary_state_check = dev_trial < velocity_threshold
    ##align the decerlation time with this velocity tracing
    dev_timing_from_analysis_window = dev_timing_from_analysis_window.astype(int)
    index_list = np.ones((len(dev_timing_from_analysis_window),), dtype=float)
    walking_length = 72
    stationary_length = 10
    p = stationary_length * "t" + walking_length * "f"
    for i in range(len(dev_timing_from_analysis_window)):
        this_dev_onset = dev_timing_from_analysis_window[i]
        bool_string = "".join(
            "t" if b else "f" for b in stationary_state_check[i, this_dev_onset:]
        )
        indices_temp = bool_string.find(p)
        if indices_temp < 0:
            indices = float("nan")
        else:
            indices = indices_temp + this_dev_onset
        index_list[i] = indices
    return index_list

def plotting_visual_guided_response(metrics_arr,stim_type,summary_df,analysis_methods,running_trial_distance_threshold,COL):
    analysis_window=analysis_methods.get("analysis_window")
    camera_fps = analysis_methods.get("camera_fps")
    stim_duration=analysis_methods.get("stim_duration")
    plot_tbt=analysis_methods.get("plotting_tbt_overview")
    color_code=analysis_methods.get("graph_colour_code")
    fig_stationary, axes_stationary = plt.subplots(
        nrows=len(stim_duration), ncols=1, figsize=(18, 7), tight_layout=True
    )
    fig_running, axes_running = plt.subplots(
        nrows=len(stim_duration), ncols=1, figsize=(18, 7), tight_layout=True
    )
    if len(stim_duration)==1:
        for i in range(len(stim_type)):
            this_duration=stim_duration[0]
            this_type=stim_type[i]
            these_velocities_running = metrics_arr[(summary_df["stim_type"] == this_type) & (summary_df['present_trial_duration']==this_duration) & (summary_df['travel_distance_isi']>running_trial_distance_threshold),:]
            these_velocities_stationary = metrics_arr[(summary_df["stim_type"] == this_type) & (summary_df['present_trial_duration']==this_duration) & (summary_df['travel_distance_isi']<running_trial_distance_threshold),:]
            # if these_velocities_stationary.shape[0]>these_velocities_running.shape[0]:
            #     num_trials_to_plot=these_velocities_stationary.shape[0]
            # else:
            #     num_trials_to_plot=these_velocities_running.shape[0]
            # COL = MplColorHelper('viridis', 0, num_trials_to_plot-1)
            num_sample=(abs(analysis_window[0])+this_duration)*camera_fps
            if plot_tbt:
                if these_velocities_stationary.shape[0]>0:
                    for this_trial in range(these_velocities_stationary.shape[0]):
                        axes_stationary.plot(np.arange(num_sample),these_velocities_stationary[this_trial,:num_sample],color=COL.get_rgb(i), linewidth=0.2)
                if these_velocities_running.shape[0]>0:
                    for this_trial in range(these_velocities_running.shape[0]):
                        axes_running.plot(np.arange(num_sample),these_velocities_running[this_trial,:num_sample],color=COL.get_rgb(i), linewidth=0.2)
                    
            else:
                if these_velocities_stationary.shape[0]>0:
                    #axes_stationary.plot(np.arange(num_sample),np.mean(these_velocities_stationary[:,:num_sample],axis=0),label=this_type,color=COL.get_rgb(i))
                    axes_stationary.plot(np.arange(num_sample),np.median(these_velocities_stationary[:,:num_sample],axis=0),label=this_type,linestyle='dotted',color=COL.get_rgb(i))
                    
                if these_velocities_running.shape[0]>0:
                    #axes_running.plot(np.arange(num_sample),np.mean(these_velocities_running[:,:num_sample],axis=0),label=this_type,color=COL.get_rgb(i))
                    axes_running.plot(np.arange(num_sample),np.median(these_velocities_running[:,:num_sample],axis=0),label=this_type,linestyle='dotted',color=COL.get_rgb(i))
        if plot_tbt==False:
            axes_stationary.legend()
            axes_running.legend()
        axes_stationary.set(xlim=[0,(abs(analysis_window[0])+max(stim_duration)+1)*camera_fps],ylim=[0,5])
        axes_running.set(xlim=[0,(abs(analysis_window[0])+max(stim_duration)+1)*camera_fps],ylim=[0,5])
    else:
        for j in range(len(stim_duration)):
            for i in range(len(stim_type)):
                this_duration=stim_duration[j]
                this_type=stim_type[i]
                these_velocities_running = metrics_arr[(summary_df["stim_type"] == this_type) & (summary_df['present_trial_duration']==this_duration) & (summary_df['travel_distance_isi']>running_trial_distance_threshold),:]
                these_velocities_stationary = metrics_arr[(summary_df["stim_type"] == this_type) & (summary_df['present_trial_duration']==this_duration) & (summary_df['travel_distance_isi']<running_trial_distance_threshold),:]
                # if these_velocities_stationary.shape[0]>these_velocities_running.shape[0]:
                #     num_trials_to_plot=these_velocities_stationary.shape[0]
                # else:
                #     num_trials_to_plot=these_velocities_running.shape[0]
                # COL = MplColorHelper('viridis', 0, num_trials_to_plot-1)
                num_sample=(abs(analysis_window[0])+this_duration)*camera_fps
                if plot_tbt:
                    if these_velocities_stationary.shape[0]>0:
                        for this_trial in range(these_velocities_stationary.shape[0]):
                            axes_stationary[j].plot(np.arange(num_sample),these_velocities_stationary[this_trial,:num_sample],color=COL.get_rgb(i), linewidth=0.2)
                    if these_velocities_running.shape[0]>0:
                        for this_trial in range(these_velocities_running.shape[0]):
                            axes_running[j].plot(np.arange(num_sample),these_velocities_running[this_trial,:num_sample],color=COL.get_rgb(i), linewidth=0.2)
                        
                else:
                    if these_velocities_stationary.shape[0]>0:
                        #axes_stationary[j].plot(np.arange(num_sample),np.mean(these_velocities_stationary[:,:num_sample],axis=0),label=this_type,color=COL.get_rgb(i))
                        axes_stationary[j].plot(np.arange(num_sample),np.median(these_velocities_stationary[:,:num_sample],axis=0),label=this_type,linestyle='dotted',color=COL.get_rgb(i))
                    if these_velocities_running.shape[0]>0:
                        #axes_running[j].plot(np.arange(num_sample),np.mean(these_velocities_running[:,:num_sample],axis=0),label=this_type,color=COL.get_rgb(i))
                        axes_running[j].plot(np.arange(num_sample),np.median(these_velocities_running[:,:num_sample],axis=0),label=this_type,linestyle='dotted',color=COL.get_rgb(i))
            # if j==0:
            #     axes_stationary[j].legend()
            #     axes_running[j].legend()
            if plot_tbt==False:
                axes_stationary[j].legend()
                axes_running[j].legend()
            axes_stationary[j].set(xlim=[0,(abs(analysis_window[0])+max(stim_duration)+1)*camera_fps])
            axes_running[j].set(xlim=[0,(abs(analysis_window[0])+max(stim_duration)+1)*camera_fps])
    return fig_stationary,fig_running,plot_tbt

def extract_events(array, min_length=10, threshold=0.5):
    """
    Extracts events from a 1D array based on given criteria.

    Parameters:
        array (np.ndarray): The input 1D array.
        min_length (int): Minimum length of the event.
        threshold (float): Minimum value for all elements in the event.

    Returns:
        List[Tuple[int, int]]: A list of tuples (start_index, end_index) for each event.
    """
    events = []
    start = None  # To track the start index of an event

    for i in range(len(array)):
        if array[i] > threshold:
            if start is None:
                start = i  # Start of a new event
        else:
            if start is not None:
                # End of an event, check its length
                if i - start >= min_length:
                    events.append((start, i))
                start = None  # Reset the start

    # Check if an event ends at the last index
    if start is not None and len(array) - start >= min_length:
        events.append((start, len(array)))

    return events


def extract_stim_locked_data(
    stim_to_analyse,
    data,
    analysis_window,
    camera_fps,
):
    # print(f"plotting data from {analysis_window[0]} to {analysis_window[1]}")
    index_points = generate_index_points(stim_to_analyse, analysis_window, camera_fps)
    if index_points.max() > len(data):
        print("index points exceed the length of the data, fill with nan in the end")
        nan_array = np.full((index_points.max() - len(data) + 10,), np.nan)
        data = np.hstack((data, nan_array))
    stim_locked_data = data[index_points]
    return stim_locked_data


def probes_optomotor_index(angular_velocity_tbt, init_frame,end_frame, camera_fps, shortened_isi):
    isi_frame = init_frame - camera_fps * shortened_isi
    opto_all_trials = np.ones((angular_velocity_tbt.shape[0]))
    opto_all_isi = np.ones((angular_velocity_tbt.shape[0]))
    ## when quantifying optomotor response, we focus on only stimulus onset so using init_frame:end_frame
    for i in range(0, angular_velocity_tbt.shape[0]):
        opto_all_trials[i] = optomotor_response_idx(
            angular_velocity_tbt[i, init_frame:end_frame]
        )
        if angular_velocity_tbt[i, isi_frame:init_frame].shape[0]>0:
            opto_all_isi[i] = optomotor_response_idx(
                angular_velocity_tbt[i, isi_frame:init_frame]
            )
        else:
            opto_all_isi[i]=np.nan
    return opto_all_trials, opto_all_isi

def find_states_with_debounce(
    x,
    thresholds=[0.002,0.005],
    min_length=100
):
    # a debounced hysteretic state machine written by chatGPT. This method should results in more sustained states against sudden changes or recording noise
    """
    Robust ON/OFF state machine with debounce against transient changes.

    Returns:
        states        : np.ndarray of "ON" / "OFF"
        on_to_off_idx : list of ON -> OFF transition indices
        off_to_on_idx : list of OFF -> ON transition indices
    """
    states = np.zeros(len(x),dtype=int)

    # ---- Initial state ----
    if x[0] > thresholds[1]:
        current_state = 1
    else:
        current_state = 0

    states[0] = current_state

    candidate_state = None
    candidate_count = 0

    # ---- State machine ----
    for i in range(1, len(x)):

        value = x[i]
        # Determine candidate state based on thresholds
        if current_state == 0 and value > thresholds[1]:
            next_candidate = 1
        elif current_state == 1 and value < thresholds[0]:
            next_candidate = 0
        else:
            next_candidate = None

        # Handle candidate accumulation
        if next_candidate is None:
            candidate_state = None
            candidate_count = 0
            states[i] = current_state
            continue

        if candidate_state == next_candidate:
            candidate_count += 1
        else:
            candidate_state = next_candidate
            candidate_count = 1

        # Confirm transition only if it persists long enough
        if candidate_count >= min_length:
            current_state = candidate_state
            #states[i] = current_state
            states[i-candidate_count:i+1]=current_state
            # if current_state == 1:
            #     off_to_on_idx.append(i-candidate_count)
            # else:
            #     on_to_off_idx.append(i-candidate_count)
            candidate_state = None
            candidate_count = 0
        else:
            states[i] = current_state
    return states#, on_to_off_idx, off_to_on_idx
def find_states_with_min_duration(signal,thresholds=[0.002,0.005],min_length=100):
    # a method to consider values over or below certian threshold for a sustained time. This method should result in more fragmented epochs
    kernel = np.ones(min_length)
    state1_binary = signal > thresholds[0]
    conv = np.convolve(state1_binary, kernel, mode="valid")
    consecutive_points = np.where(conv >= min_length)[0]
    if consecutive_points.shape[0]>0:
        state1_starts = consecutive_points[np.insert(np.diff(consecutive_points)>=min_length, 0, True)]
    else:
        state1_starts=np.nan
    kernel = np.ones(min_length)
    state2_binary = signal < thresholds[1]
    conv = np.convolve(state2_binary, kernel, mode="valid")
    consecutive_points = np.where(conv >= min_length)[0]    
    if consecutive_points.shape[0]>0:
        state2_starts = consecutive_points[np.insert(np.diff(consecutive_points)>=min_length, 0, True)]
    else:
        state2_starts=np.nan
    return state1_starts,state2_starts

def identify_behavioural_states(travel_distance_fbf,angular_velocity,filter_method,camera_fps,consecutive_duration=[0.25,0.25],smooth_window_length=51,skip_smoothing=None,walk_thresholds=None,turn_thresholds=None):
    consecutive_length=[camera_fps*consecutive_duration[0],camera_fps*consecutive_duration[1]]
    if skip_smoothing==None:
        if filter_method in ["sg_filter","rolling_median"]:
            skip_smoothing=True
        else:
            skip_smoothing=False
    if skip_smoothing==True:      
        speed_signal =  travel_distance_fbf
        turn_signal =  angular_velocity
    else:
        speed_signal =  medfilt(travel_distance_fbf, int(smooth_window_length))
        turn_signal =  medfilt(angular_velocity, int(smooth_window_length))
    if type(walk_thresholds)==list and type(turn_thresholds)==list:
        print("use customised threshold")
        pass
    else:
        if skip_smoothing==True and filter_method=='rolling_median':
            walk_thresholds=[0.001,0.005]
            turn_thresholds=[0.1,0.2]
            warnings.warn("rolling_median filter is not applied to speed. consider using use medfilt for that")
        elif skip_smoothing==False and filter_method=='rolling_median':
            walk_thresholds=[0.005,0.008]#stop here
            turn_thresholds=[0.2,0.3]
        elif skip_smoothing==True and filter_method=="sg_filter":
            walk_thresholds=[0.001,0.003]
            turn_thresholds=[0.25,0.5]
            warnings.warn("sg filter is not applied to angular velocity. consider using use medfilt for that")
        elif skip_smoothing==False and filter_method=="sg_filter":
            walk_thresholds=[0.001,0.003]
            turn_thresholds=[0.25,0.5]
        elif skip_smoothing==True:
            warnings.warn("no smoothing method is applied before classified state. The result will be susceptible to noise")
            walk_thresholds=[0.01,0.02]
            turn_thresholds=[0.25,0.5]
        else:
            walk_thresholds=[0.003,0.01]
            turn_thresholds=[0.2,0.5]

    walk_states=find_states_with_debounce(speed_signal,thresholds=walk_thresholds,min_length=consecutive_length[0])
    turn_states=find_states_with_debounce(abs(turn_signal),thresholds=turn_thresholds,min_length=consecutive_length[1])
    turn_cw=find_states_with_debounce(turn_signal,thresholds=turn_thresholds,min_length=consecutive_length[1])
    turn_ccw=find_states_with_debounce(turn_signal*-1,thresholds=turn_thresholds,min_length=consecutive_length[1])
    tmin_max=[20000,30000]#-1 for the entire trace or specify a frame number
    #tmin_max=[0,-1]#-1 for the entire trace or specify a frame number
    run_time=np.arange(speed_signal.shape[0])
    if tmin_max[1]==-1:
        section_length=speed_signal.shape[0]-2
    else:
        section_length=tmin_max[1]-tmin_max[0]-1

    frame_no=run_time[tmin_max[0]:tmin_max[1]]
    fig, ((ax3,ax4),(ax5,ax6)) = plt.subplots(
            nrows=2, ncols=2, figsize=(18, 7), tight_layout=True
        )    
    ax3.scatter(frame_no,speed_signal[tmin_max[0]:tmin_max[1]],s=0.1)
    #ax3.scatter(frame_no,speed_signal,s=0.1)
    ax3.set(title='after medfilt speed')
    ax3.axhline(y=walk_thresholds[1],xmin=frame_no[0],xmax=frame_no[section_length],linewidth=3,color='r')
    ax3.axhline(y=walk_thresholds[0],xmin=frame_no[0],xmax=frame_no[section_length],linewidth=3,color='k')
    ax5.scatter(np.where(walk_states[tmin_max[0]:tmin_max[1]]==0)[0],np.zeros((np.where(walk_states[tmin_max[0]:tmin_max[1]]==0)[0].shape)),s=0.1)
    ax5.scatter(np.where(walk_states[tmin_max[0]:tmin_max[1]]==1)[0],np.ones((np.where(walk_states[tmin_max[0]:tmin_max[1]]==1)[0].shape)),s=0.1)
    run_time=np.arange(turn_signal.shape[0])
    if tmin_max[1]==-1:
        section_length=turn_signal.shape[0]-2
    else:
        section_length=tmin_max[1]-tmin_max[0]-1
    frame_no=run_time[tmin_max[0]:tmin_max[1]]
    ax4.scatter(frame_no,abs(turn_signal[tmin_max[0]:tmin_max[1]]),s=0.1)
    #ax4.scatter(frame_no,abs(turn_signal),s=0.1)
    ax4.set(title='after medfilt abs rotational velocity')
    ax4.axhline(y=turn_thresholds[1],xmin=frame_no[0],xmax=frame_no[section_length],linewidth=3,color='r')
    ax4.axhline(y=turn_thresholds[0],xmin=frame_no[0],xmax=frame_no[section_length],linewidth=3,color='k')
    ax6.scatter(np.where(turn_states[tmin_max[0]:tmin_max[1]]==0)[0],np.zeros((np.where(turn_states[tmin_max[0]:tmin_max[1]]==0)[0].shape)),s=0.1)
    ax6.scatter(np.where(turn_states[tmin_max[0]:tmin_max[1]]==1)[0],np.ones((np.where(turn_states[tmin_max[0]:tmin_max[1]]==1)[0].shape)),s=0.1)    
    return walk_states,turn_states,turn_cw,turn_ccw,fig


def classify_trial_type(summary_df, turning_threshold, running_speed_threshold):
    if summary_df.stim_type.dtype == "object":
        print(
            "classify the behaviour into turning, running, straightwalk, turning_walk"
        )
    else:
        print(
            "classify the behaviour into turning, running, straightwalk, turning_walk + follow and against trials"
        )
        summary_df["optoxstim"] = summary_df["opto_index"] * summary_df["stim_type"]

    running_trials = summary_df.loc[
        summary_df["travel_distance"] > running_speed_threshold
    ]
    stationary_trials = summary_df.loc[
        summary_df["travel_distance"] < running_speed_threshold
    ]
    if 'opto_index' in summary_df.columns:
        turning_trials = summary_df.loc[abs(summary_df["opto_index"]) > turning_threshold]

        turning_walk_trials = summary_df.loc[
            (summary_df["travel_distance"] > running_speed_threshold)
            & (abs(summary_df["opto_index"]) > turning_threshold)
        ]
        straight_walk_trials = summary_df.loc[
            (summary_df["travel_distance"] > running_speed_threshold)
            & (abs(summary_df["opto_index"]) < turning_threshold)
        ]
    else:
       turning_trials=None
       turning_walk_trials=None
       straight_walk_trials=None
    if "optoxstim" in summary_df:
        follow_trials = turning_walk_trials.loc[turning_walk_trials["optoxstim"] < 0]
        against_trials = turning_walk_trials.loc[turning_walk_trials["optoxstim"] > 0]
        spontaneous_trials = turning_walk_trials.loc[
            turning_walk_trials["optoxstim"] == 0
        ]
        tmp = follow_trials.groupby("stim_type").count()
        follow_count_coherence = tmp.index.values
        follow_count = tmp.values[:, 0]
    return running_trials,stationary_trials,straight_walk_trials,turning_trials
def plotting_example_trajectory(x_all,y_all,start_frame=0,end_frame=None,start_marker=[],end_marker=[],yaw_angular_velocity=[]):
    markers_size=400
    camera_fps=100
    trajectory_size=60
    colormap_name = "viridis"
    COL = MplColorHelper(colormap_name, 0, 6)
    visualise_section_by_colour=True
    fig, ax= plt.subplots(1, 1, figsize=(27, 7), tight_layout=True)
    fig2,(ax2,ax3)= plt.subplots(2, 1, figsize=(27, 7),sharex=True,tight_layout=True)
    if end_frame==None:
        ax.scatter(y_all[start_frame:],x_all[start_frame:],c=np.arange(len(y_all[start_frame:])), marker=".", alpha=0.5)
    else:
        if visualise_section_by_colour==True:
            ax.scatter(y_all[start_frame:start_marker[0]],x_all[start_frame:start_marker[0]],s=trajectory_size,c='grey',marker=".", alpha=0.5)
            ax.scatter(y_all[start_marker[0]:end_marker[0]],x_all[start_marker[0]:end_marker[0]],s=trajectory_size,c='blue', marker=".", alpha=0.5)
            ax.scatter(y_all[end_marker[0]:start_marker[1]],x_all[end_marker[0]:start_marker[1]],s=trajectory_size,c='grey',marker=".", alpha=0.5)
            ax.scatter(y_all[start_marker[1]:end_marker[1]],x_all[start_marker[1]:end_marker[1]],s=trajectory_size,c='red', marker=".", alpha=0.5)
            ax.scatter(y_all[end_marker[1]:end_frame],x_all[end_marker[1]:end_frame],s=trajectory_size,c='grey',marker=".", alpha=0.5)
            #travel_distance_tbt=euclidean_distance(x_all[start_frame:start_marker[0]],y_all[start_frame:start_marker[0]])
            #velocity_arr = travel_distance_tbt * camera_fps
            angular_velocity_interest=yaw_angular_velocity[start_frame:start_marker[0]]
            #delta_angular_velocity=np.diff(angular_velocity_interest)
            delta_angular_velocity=np.gradient(angular_velocity_interest, 1/camera_fps)
            ax2.plot(np.arange(angular_velocity_interest.shape[0])/camera_fps,angular_velocity_interest,color='grey')
            ax3.plot(np.arange(delta_angular_velocity.shape[0])/camera_fps,delta_angular_velocity,color='grey')
            angular_velocity_interest=yaw_angular_velocity[start_marker[0]:end_marker[0]]
            delta_angular_velocity=np.gradient(angular_velocity_interest, 1/camera_fps)
            
            relative_start=start_marker[0]-start_frame
            ax2.plot(np.arange(relative_start,angular_velocity_interest.shape[0]+relative_start)/camera_fps,angular_velocity_interest,color='blue')
            ax3.plot(np.arange(relative_start,delta_angular_velocity.shape[0]+relative_start)/camera_fps,delta_angular_velocity,color='blue')
            angular_velocity_interest=yaw_angular_velocity[end_marker[0]:start_marker[1]]
            delta_angular_velocity=np.gradient(angular_velocity_interest, 1/camera_fps)
            relative_start=end_marker[0]-start_frame
            ax2.plot(np.arange(relative_start,angular_velocity_interest.shape[0]+relative_start)/camera_fps,angular_velocity_interest,color='grey')
            ax3.plot(np.arange(relative_start,delta_angular_velocity.shape[0]+relative_start)/camera_fps,delta_angular_velocity,color='grey')
            angular_velocity_interest=yaw_angular_velocity[start_marker[1]:end_marker[1]]
            delta_angular_velocity=np.gradient(angular_velocity_interest, 1/camera_fps)
            relative_start=start_marker[1]-start_frame
            ax2.plot(np.arange(relative_start,angular_velocity_interest.shape[0]+relative_start)/camera_fps,angular_velocity_interest,color='red')
            ax3.plot(np.arange(relative_start,delta_angular_velocity.shape[0]+relative_start)/camera_fps,delta_angular_velocity,color='red')
            angular_velocity_interest=yaw_angular_velocity[end_marker[1]:end_frame]
            delta_angular_velocity=np.gradient(angular_velocity_interest, 1/camera_fps)
            relative_start=end_marker[1]-start_frame
            ax2.plot(np.arange(relative_start,angular_velocity_interest.shape[0]+relative_start)/camera_fps,angular_velocity_interest,color='grey')
            ax3.plot(np.arange(relative_start,delta_angular_velocity.shape[0]+relative_start)/camera_fps,delta_angular_velocity,color='grey')
            ax2.set_ylim(-5.4, 5.4)
            #ax2.set_yticks([-4,0,4])
            ax2.set(yticks=[-4, 0, 4])
            ax2.set_ylabel(r'Angular velocity (rad s$^{-1}$)')
            ax3.set_ylim(-350,350)
            #ax3.set_yticks([-300,0,300])
            ax3.set(yticks=[-300, 0, 300])
            ax3.set_ylabel(r'Angular acceleration (rad s$^{-2}$)')
            ax3.set_xlabel("Time (s)")
            fig2.savefig("angular_velocity_example_trials.png")
        else:
            '''
            cols = np.linspace(0,1,len(x_all[start_frame:end_frame]))
            points = np.array([x_all[start_frame:end_frame], y_all[start_frame:end_frame]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='viridis')
            lc.set_array(cols)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            fig.colorbar(line,ax=ax)
            '''
            ax.scatter(y_all[start_frame:end_frame],x_all[start_frame:end_frame],s=trajectory_size,color=np.arange(len(y_all[start_frame:end_frame])), marker=".", alpha=0.5)
        ax.axis('off')
    if type(start_marker) == list:
        if len(start_marker) > 0:
            for i in range(len(start_marker)):
                ax.scatter(y_all[start_marker[i]],x_all[start_marker[i]],s=markers_size,color=COL.get_rgb(6), marker="*",edgecolors='black')
    else:
        if start_marker > 0:
            ax.scatter(y_all[start_marker],x_all[start_marker],s=markers_size,color=COL.get_rgb(6), marker="*",edgecolors='black')

    if type(end_marker) == list:
        if len(end_marker) > 0:
            for i in range(len(end_marker)):
                ax.scatter(y_all[end_marker[i]],x_all[end_marker[i]],s=markers_size,color=COL.get_rgb(0), marker="*",edgecolors='black')
    else:
        if end_marker > 0:
            ax.scatter(y_all[end_marker],x_all[end_marker],s=markers_size,color=COL.get_rgb(0), marker="*",edgecolors='black')
    return fig


def generate_index_points(stim_to_analyse, analysis_window, camera_fps):
    window_length = analysis_window[1] - analysis_window[0]
    interval_onset = (
        stim_to_analyse
        + np.ones(len(stim_to_analyse), dtype=int) * analysis_window[0] * camera_fps
    )
    analysie_window_tile = np.tile(
        np.arange(window_length * camera_fps), (len(interval_onset), 1)
    )
    index_points = (
        np.array(interval_onset).reshape(len(interval_onset), 1) + analysie_window_tile
    )
    return index_points


def optomotor_response_idx(ev_dat):
    opt_idx = (
        np.nansum(abs(ev_dat[ev_dat > 0])) - np.nansum(abs(ev_dat[ev_dat < 0]))
    ) / (np.nansum(abs(ev_dat[ev_dat < 0])) + np.nansum(abs(ev_dat[ev_dat > 0])))
    return opt_idx



def preprocess_tracking_data(df,filtering_method,yaw_axis,smooth_window_length):
    if filtering_method == "sg_filter":
        x_all = savgol_filter(df.loc[:, ["intergrated x position"]].values[:,0], smooth_window_length, 3, axis=0)
        y_all = savgol_filter(df.loc[:, ["intergrated y position"]].values[:,0], smooth_window_length, 3, axis=0)
        yaw_angular_velocity =df.loc[:, [f"delta rotation vector lab {yaw_axis}"]].values[:,0]
    elif filtering_method == "rolling_median":
        x_all =df.loc[:, ["intergrated x position"]].rolling(smooth_window_length).median().values
        y_all = df.loc[:, ["intergrated y position"]].rolling(smooth_window_length).median().values
        yaw_angular_velocity =df.loc[:, [f"delta rotation vector lab {yaw_axis}"]].rolling(smooth_window_length).median().values
        x_all=bfill(x_all[:,0])
        y_all=bfill(y_all[:,0])
        yaw_angular_velocity=bfill(yaw_angular_velocity[:,0])
    else:
        x_all = df.loc[:, ["intergrated x position"]].values[:,0]
        y_all = df.loc[:, ["intergrated y position"]].values[:,0]
        yaw_angular_velocity = df.loc[:, [f"delta rotation vector lab {yaw_axis}"]].values[:,0]    
    
    return x_all,y_all,yaw_angular_velocity*-1 # multiply by -1 to get animal's angular velocity. Otherwise, it is the angular velocity of the ball. After multiplying by -1, positive value means animals making cw turn.

def main(thisDir, json_file):
    if type(thisDir)==str:
        thisDir = Path(thisDir)
    # Figuring out path and directory
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    BODY_LENGTH=4
    # Load analysis methods
    load_previous_methods=analysis_methods.get("load_previous_methods",False)
    if load_previous_methods:
        previous_methods_file=find_file(thisDir, "analysis_methods_dictionary_backup.json")
        if previous_methods_file!=None:
            with open(previous_methods_file, "r") as f:
                print(f"load analysis methods from previous file {previous_methods_file}")
                previous_analysis_methods = json.loads(f.read())
            analysis_methods.update(previous_analysis_methods)
        else:
            print("previous analysis methods file is not found. Use the current one.")
    ## Here to load environmental variables about the setup
    exp_place = analysis_methods.get("exp_place")
    monitor_fps = analysis_methods.get("monitor_fps")
    camera_fps = analysis_methods.get("camera_fps",100)
    trackball_radius=analysis_methods.get("trackball_radius",5)## track ball radius in (cm)
    fictrac_posthoc_analysis = analysis_methods.get("fictrac_posthoc_analysis")
    temperature_data_options = analysis_methods.get("temperature_data_options")
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset")
    save_output = analysis_methods.get("save_output")
    ## Here to load environmental variables about the stimulus
    stim_duration = analysis_methods.get("stim_duration")
    interval_duration= analysis_methods.get("interval_duration")
    visual_paradigm_name = analysis_methods.get("experiment_name")
    align_with_isi_onset = analysis_methods.get("align_with_isi_onset")
    create_null_null_condition=analysis_methods.get("create_null_null_condition",False)
    prestim_duration=analysis_methods.get("prestim_duration")

    ## Here to load environmental variables about the behaviour
    mark_jump_as_nan = analysis_methods.get("mark_jump_as_nan", False)
    duration_for_optomotor_index=analysis_methods.get("duration_for_optomotor_index",0)
    num_step_per_decision=analysis_methods.get("num_step_per_decision",6)
    yaw_axis=analysis_methods.get("yaw_axis","z").lower()# define which axis to use to examine animal's rotational velocity
    filtering_method=analysis_methods.get("filtering_method")
    smooth_window_length = round(0.5*camera_fps)
    smooth_window_length = smooth_window_length if np.mod(smooth_window_length, 2) == 1 else smooth_window_length + 1 #must be an odd number for median filter and sg filter

    # Here to load environmental variables about plotting
    colormap_name = "coolwarm"
    sm = cm.ScalarMappable(cmap=colormap_name)
    # Here to allocate files
    pd_start="meta"
    PD_file_pattern1 = "lux*.csv"
    PD_file_pattern2 = f"{pd_start}*.csv"
    PD_file_directory = find_file(thisDir, PD_file_pattern2,PD_file_pattern1)
    if PD_file_directory is not None:
        if PD_file_directory.stem.startswith("meta1"):
            print("you probably use camera 1 to run this experiment. The default camera fps is 144 Hz")
            analysis_methods.update({"camera_fps":144})
        elif PD_file_directory.stem.startswith("meta2") or PD_file_directory.stem.startswith("meta"):
            print("you probably use camera 2 to run this experiment. The default camera fps is 100 Hz")
            analysis_methods.update({"camera_fps":100})
    trial_file_pattern = "trial*.csv"
    trial_file_directory = find_file(thisDir, trial_file_pattern)
    tracking_file_pattern = "tracking*.csv"
    tracking_file_directory = find_file(thisDir, tracking_file_pattern)
    database_file_pattern ="database*.parquet.gzip"
    #database_file_pattern ="database*.pickle"
    database = find_file(thisDir, database_file_pattern)
    temperature_file_pattern = "temperature_humidity.npy.gz"
    this_temp_directory = find_file(thisDir, temperature_file_pattern)

    ##here to load tracking data
    if database is None and fictrac_posthoc_analysis is False:
        df=pd.read_csv(tracking_file_directory,skiprows=0)
        if df.shape[1]==4:
            columns_name=["intergrated x position","intergrated y position",f"delta rotation vector lab {yaw_axis}","pd_phase"]
        elif df.shape[1]==5:
            columns_name=["intergrated x position","intergrated y position",f"delta rotation vector lab {yaw_axis}","frame_count","pd_phase"]
        df.columns=columns_name
        df[["intergrated x position","intergrated y position"]]=df[["intergrated x position","intergrated y position"]]*trackball_radius
        df[f"delta rotation vector lab {yaw_axis}"] = df[f"delta rotation vector lab {yaw_axis}"] * camera_fps
    elif database_file_pattern.endswith(".parquet.gzip"):
        df=pd.read_parquet(database)
    else:
        df=pd.read_pickle(database)

    x_all,y_all,yaw_angular_velocity=preprocess_tracking_data(df,filtering_method,yaw_axis,smooth_window_length)
    travel_distance_fbf = euclidean_distance(x_all,y_all)
    walk_states,turn_states,turn_cw,turn_ccw,fig=identify_behavioural_states(travel_distance_fbf,yaw_angular_velocity,filtering_method,camera_fps,consecutive_duration=[2,0.5],smooth_window_length=smooth_window_length,skip_smoothing=False)
    plot_name = f"behavioural_states_classification.png"
    if save_output == True:
        fig.savefig(thisDir / plot_name)
    ## here write an additional function to classify stationary, moving and total travel distance
    if mark_jump_as_nan:
        s2w,w2s=find_states_with_min_duration(travel_distance_fbf,thresholds=[1,10], min_length=20)
        events = extract_events(travel_distance_fbf, min_length=20)
        ##use fft to classify trials where animals jump. So far this method can be used to detect some obvious jumps but there might be some miss detecting
        auc = np.zeros(len(events))
        for i in range(len(events)):
            event_to_analyse = np.transpose(
                travel_distance_fbf[events[i][0] : events[i][1]]
            )[0]
            fft_power, freq_axis = calculate_fft(
                event_to_analyse,
                camera_fps,
            )
            N = event_to_analyse.size
            amplitude = 2.0 / N * np.abs(fft_power[0 : N // 2])
            auc[i] = amplitude[0]
            if (
                amplitude[0] > 2
            ):  # need to use signal detection theory to find a good threshold in the future;
                # 2 is a safer threshold, which would includes false negative cases.
                travel_distance_fbf[events[i][0] : events[i][1]] = np.nan
    else:
        pass
    travel_distance_whole_session = np.nansum(travel_distance_fbf)/camera_fps
    if analysis_methods.get("plotting_trajectory") == True:
        #fig=plotting_example_trajectory(x_all,y_all)
        fig=plotting_example_trajectory(x_all,y_all,start_frame=154892,end_frame=163473,start_marker=[155901,159436],end_marker=[158426,161961],yaw_angular_velocity=yaw_angular_velocity)
        fig.savefig(thisDir / "trajectory_plot_example_trials.png")
    if analysis_methods.get("analyse_spontaneous_activity")==True:
        
        return

    # Here to load time based on one of the files created by bonsai such as PD file or events files
    if PD_file_directory:   
        file_name = PD_file_directory.stem
        if pd_start == "meta":
            _, this_date_hour, this_min, this_sec = file_name.split("_")
            exp_time = f"{this_date_hour}_{this_min}_{this_sec}"
        else:
            _, exp_time = re.split(
                rf"{re.escape(pd_start)}\d_",
                file_name,
                maxsplit=1
            )
        exp_date, exp_hour = exp_time.split("T", maxsplit=1)
        time_in_pd_timestamp = pd.to_datetime(exp_time, format="%Y-%m-%dT%H_%M_%S")
    else:
        [exp_date, exp_hour] = trial_file_directory.stem.split("events")[1].split("T")
        exp_time = exp_date + " " + exp_hour
        time_in_pd_timestamp = pd.to_datetime(exp_time, format="%Y-%m-%d %H_%M_%S")
    # Here to load temperature data if possible
    tem_df_this_exp = []
    if temperature_data_options == None: 
        print("Temperature is not recorded in this experiment.")
    elif temperature_data_options.lower() == "trial":# this means temperature data is stored together with the trial logging file
        print("Temperature is recorded together with trial information.")
    else:
        if this_temp_directory and overwrite_curated_dataset == False:
            f = gzip.GzipFile(this_temp_directory, "r")
            tem_df_this_exp = np.load(f)
        ### re-load temerature data if the user wants to overwrite the curated dataset or if temperature data is not available
        elif temperature_data_options.lower().startswith("bon"):
            temperature_file_pattern = "lux*.csv"
            termperature_pattern_meta="meta*.csv"
            led_files = find_file(thisDir, temperature_file_pattern,termperature_pattern_meta)
            if led_files!=None:
                tmp = pd.read_csv(led_files, delimiter=",", header=0, index_col=None)
                if "Temperature" in tmp.columns:
                    tem_df_this_exp = np.column_stack(
                        (
                            tmp["Temperature"].values,
                            tmp["Humidity"].values,
                        )
                    )
                else:
                    print("work in progress. Needs to organise this pandas dataframe")
        else:
            if temperature_data_options.lower() == "el_usb":
                temperature_file_pattern = "temperature_humidity*.txt"
            if temperature_data_options == "DL220THP":
                temperature_file_pattern = "DL220*.csv"
            this_temp_directory = find_file(thisDir, temperature_file_pattern)
            try:
                tem_df = load_temperature_data(txt_path=this_temp_directory)
                start_time_tem = find_nearest(tem_df.index, time_in_pd_timestamp)
                print(
                    tem_df[tem_df.index == start_time_tem]
                )  # find the start of EL-USB data that might be used for further data analysis. For example, oversampling to align with existing data size
                tem_df_after_exp = tem_df[tem_df.index > start_time_tem]
                if len(tem_df_after_exp) > 0:
                    frequency_milisecond = int(1000 / camera_fps)
                    tem_df_after_exp = tem_df_after_exp.resample(
                        f"{frequency_milisecond}L"
                    ).interpolate()
                    tem_df_this_exp = tem_df_after_exp[: df.shape[0]].values[:, 0:2]
                else:
                    print("Unable to find the corresponding time stamp")
            except:
                print("raw temperature file is not found")
        if save_output == True and len(tem_df_this_exp) > 0:
            np_file_name = thisDir / "temperature_humidity.npy.gz"
            f = gzip.GzipFile(np_file_name, "w")
            np.save(file=f, arr=tem_df_this_exp)
            f.close()
        else:
            print("Nothing is saved in the debug mode. Move to the next animal")

    stim_info = pd.read_csv(trial_file_directory)
    # if file_name == "lux4_2024-12-09T12_02_03":
    #     stim_info = stim_info.iloc[:336]
    stim_info, stim_type = sorting_trial_info(
        stim_info, analysis_methods,exp_date)
    stimulus_timestamp = stim_info["ts"].to_numpy()
    ### define analysis window
    # Use analysis window to slice recordings into sections centered the onset of the stimulus. If analysis window is not defiend, then use interval_duration and stim duration to construct that.
    # While duration consists of more than two int, use the largest one to set analysis window
    if analysis_methods.get("analysis_window") == None:
        analysis_window=define_analysis_window(stim_duration,interval_duration)
    else:
        analysis_window = analysis_methods.get("analysis_window")
        if visual_paradigm_name == "sweeploom":
            presence_duration = 1
            velocity_threshold = 5
    ### analyse LED signal to align the onset and offset of stimuli with video frames
    if fictrac_posthoc_analysis:
        stim_onset, isi_onset = analyse_led_signal(thisDir, analysis_methods, stim_info)
        if visual_paradigm_name in ["gratings","sweeploom","conflict","looming","sweeping","densities","choices"]:
            if 'PreMovDuration' in stim_info.columns:
                if stim_info['PreMovDuration'].unique()!=0 and exp_place!='MatrexVR':
                    if stim_onset.shape[0]%2==1:
                        stim_onset=stim_onset[:-1]
                    if isi_onset.shape[0]%2==1:
                        isi_onset=isi_onset[:-1]

                    stim_appearance_onset=stim_onset[::2]
                    stim_motion_stop_onset=stim_onset[1::2]
                    stim_onset=isi_onset[::2]
                    isi_onset=isi_onset[1::2]
                    print("stimulus duration + 0.5 sec")
                    print(isi_onset-stim_onset)
            stim_info["stim_onset"] = stim_onset
        elif visual_paradigm_name =='combo':
            num_TTL_1stset=stim_info[stim_info['stim_type']=='combo_colour'].shape[0]
            TTL_2ndset1=stim_onset[num_TTL_1stset:]
            TTL_2ndset2=isi_onset[num_TTL_1stset:]
            stim_appearance_onset=TTL_2ndset1[::2][:stim_info[stim_info['stim_type']=='combo_size'].shape[0]]
            stim_motion_stop_onset=TTL_2ndset1[1::2]
            stim_onset_frame=np.concatenate((stim_onset[:num_TTL_1stset], TTL_2ndset2[::2][:TTL_2ndset2[1::2].shape[0]]))
            isi_onset_frame=np.concatenate((isi_onset[:num_TTL_1stset], TTL_2ndset2[1::2]))
            isi_onset=isi_onset_frame
            stim_info["stim_onset"] = stim_onset=stim_onset_frame
        else:
            transition_onset = np.ones((len(stim_onset) + len(isi_onset),), dtype=int)
            transition_onset[1::2] = stim_onset
            transition_onset[::2] = isi_onset
            if -1 in stim_info['Duration'].unique():
                duration_array = np.ones((len(stim_onset) + len(isi_onset),), dtype=int)* np.nan
                stim_duration_array=np.round((isi_onset[1:]-stim_onset[:-1])/camera_fps)
                isi_duration_array=np.round((stim_onset-isi_onset)/camera_fps)
                duration_array[1:-1:2] = stim_duration_array
                duration_array[::2] = isi_duration_array
                stim_info["Duration"] = duration_array

            stim_info["stim_onset"] = transition_onset
    elif "pd_phase" in df.columns:##this was created for closed-loop choices assay
        transition_values=np.diff(df['pd_phase'].values)
        if stim_info['PreMovDuration'].unique()==0:
            transition_onsets=np.where(transition_values)[0]
            stim_onset=transition_onsets[::2]
            isi_onset=transition_onsets[1::2]
            stim_info["stim_onset"] = stim_onset
        else:
            transition_to_on=np.where(transition_values==1)[0]
            transition_to_off=np.where(transition_values==-1)[0]
            stim_appearance_onset=transition_to_on[::2]
            stim_motion_stop_onset=transition_to_on[1::2]
            stim_onset=transition_to_off[::2]
            isi_onset=transition_to_off[1::2]
            stim_info["stim_onset"] = stim_onset

    elif df["timestamp"][1] != df["timestamp"][2]:
        # compare the timestamp between event file and dat file
        data_timestamp = df.loc[:, "timestamp"]
        stimulus_timestamp_b = np.tile(stimulus_timestamp, (len(data_timestamp), 1))
        data_timestamp_b = np.tile(data_timestamp, (len(stimulus_timestamp), 1))
        timestamp_diff = data_timestamp_b > np.transpose(stimulus_timestamp_b)
        stim_onset = np.where(np.roll(timestamp_diff, 1, axis=1) != timestamp_diff)[1][
            1::2
        ]
        isi_onset=stim_onset+(stim_info['Duration'].values+presence_duration*2)*camera_fps
        stim_info["stim_onset"] = stim_onset
    else:
        warnings.warn(
            "not support using tracking_info timestamp anymore. Use older version for this analysis"
        )
        return
    ##probably need to add some function here to deal with the situation when isi_onset_log is True
    log_stim_ISI_duration=True### cam not remember what this option means. Remove it in the next update.
    if align_with_isi_onset:
        ##this was designed for RDK experiment but it is not yet used
        stim_to_analyse = stim_info.loc[::2, "stim_onset"]
        ISI_duration = np.array(
                stim_info["Duration"][stim_info.loc[1::2, "stim_onset"].index]
            )
        if log_stim_ISI_duration:
            preStim_ISI_duration = np.append(
                analysis_methods.get("prestim_duration"), ISI_duration[:-1]
            )
        else:
            preStim_ISI_duration=stim_info
    else:
        if visual_paradigm_name in ["gratings","sweeploom","conflict","looming","sweeping","choices","densities","combo"]:
            stim_to_analyse = stim_info["stim_onset"]
            if "ISI" in stim_info.columns:
                if type(analysis_methods.get("interval_duration")) == list and any(stim_info["ISI"].values>0):## for the situation where ISI is pre-defiend
                    ISI_duration = stim_info["ISI"].values
                elif any(stim_info["ISI"].values<0):## for the situation where ISI is not pre-defined
                    if 'stim_appearance_onset' in locals():## for the situation there is stimulus appearance and motion are seperated action
                        if visual_paradigm_name=="combo":
                            ISI_duration=np.concatenate((stim_onset[1:num_TTL_1stset+1]-isi_onset[:num_TTL_1stset],stim_appearance_onset[1:]-isi_onset[num_TTL_1stset:-1]))
                        else:
                            ISI_duration=np.round((stim_appearance_onset[1:]-isi_onset[:-1])/camera_fps)
                    else:
                        ISI_duration=np.round((stim_onset[1:]-isi_onset[:-1])/camera_fps)
                    ISI_duration=np.append(ISI_duration, np.nan)
                else:## for the situation where ISI is pre-defined as a fixed value
                    ISI_duration = np.repeat(
                        analysis_methods.get("interval_duration"), stim_to_analyse.shape[0]
                    )
            elif 'isi_onset' in locals():
                ISI_duration=np.round((stim_onset[1:]-isi_onset[:-1])/camera_fps)#I am curious which one uses here besides densities
            else:
                stim_isi_duration=np.round(np.diff(stim_onset)/camera_fps)
                ISI_duration=np.append(stim_isi_duration, [np.nan])-stim_info['Duration'].values-presence_duration*2
            num_stim=stim_info.shape[0]
            analysis_methods.update({"prestim_duration":prestim_duration})
            if ISI_duration.shape[0]==num_stim:### this mean the duration before stimulus. In a typical workflow, the duration of post-stimulus interval was logged
                preStim_ISI_duration = np.append(
                    analysis_methods.get("prestim_duration"), ISI_duration[:-1]
                )
            else:### In the case of density, the ISI is neither logged nor specified in the trial information so that the duration of post-stimulus interval would be mixed with the end of experiment, usually discard to avoid confusion.
                preStim_ISI_duration = np.append(
                    analysis_methods.get("prestim_duration"), ISI_duration
                )
            
        else:
            stim_to_analyse = stim_info.loc[1::2, "stim_onset"]
            if log_stim_ISI_duration:
                preStim_ISI_duration = np.array(
                    stim_info["Duration"][stim_info.loc[::2, "stim_onset"].index]
                )
            else:
                preStim_ISI_duration=stim_info
            num_stim=int(stim_info.shape[0]/2)
    if create_null_null_condition:
        print("create null null condition if previous ISI is 40 sec. Set the 20th ISI as the start of null null trial")
        stim_info['ISI']=ISI_duration
        index_to_insert=stim_to_analyse[ISI_duration==40].index
        null_null_starting_frame=np.floor((stim_to_analyse[ISI_duration==40].values+stim_to_analyse[stim_to_analyse[ISI_duration==40].index+1].values)/2)
        null_columns = stim_info.loc[index_to_insert+1]
        null_columns['stim_onset']=null_null_starting_frame.astype(int)
        null_columns['stim_type']='null_null'
        tmp=pd.concat([stim_info,null_columns])
        stim_info = tmp.sort_values(by='stim_onset').reset_index(drop=True)
        stim_to_analyse=stim_info['stim_onset']
        stim_onset=stim_to_analyse.values
        stim_info.loc[stim_info['stim_type']=='null_null','ISI']=np.nan
        num_stim=stim_to_analyse.shape[0]
        ISI_duration=stim_info['ISI'].values
        preStim_ISI_duration = np.append(analysis_methods.get("prestim_duration"), ISI_duration[:-1])
        isi_onset=np.insert(isi_onset, index_to_insert+1,-1)
        stim_type=['black_black','glocust_glocust','black_glocust','glocust_black','black_null', 'glocust_null', 'null_black', 'null_glocust','null_null']

    stim_name_arr = np.array(stim_info["stim_type"][stim_to_analyse.index])
    if any(stim_info["Duration"].values<0):
        if visual_paradigm_name=='coherence':
            present_duration_arr = np.round((isi_onset[1:]-stim_onset[:-1])/camera_fps)
            present_duration_arr=np.append(present_duration_arr, np.nan)
        else:
            present_duration_arr = np.round((isi_onset-stim_onset)/camera_fps)
    else:
        present_duration_arr = np.array(stim_info["Duration"][stim_to_analyse.index])
    (
        summary_list,
        angular_velocity_tbt,
        travel_distance_tbt,
        init_frame,
        end_frame,
    ) = analysis_stim_evoked_response_combo(
        stim_to_analyse,
        stim_info,
        yaw_angular_velocity,
        travel_distance_fbf,
        tem_df_this_exp,
        analysis_methods
    )
    velocity_arr = travel_distance_tbt * camera_fps  # over 1.0 is probably a jump
    cum_travel_distance = np.nancumsum(velocity_arr, axis=1)/camera_fps
    heading_angle = np.nancumsum(angular_velocity_tbt, axis=1)/camera_fps
    heading_at_stim=heading_angle[:,init_frame]
    heading_prev_stim=heading_angle[:,:init_frame]-np.reshape(np.repeat(heading_at_stim,init_frame), (num_stim, init_frame))
    heading_post_stim=heading_angle[:,init_frame:]-np.reshape(np.repeat(heading_at_stim,end_frame-init_frame), (num_stim, end_frame-init_frame))
    distance_at_stim=cum_travel_distance[:,init_frame]
    cum_distance_prev_stim=cum_travel_distance[:,:init_frame]-np.reshape(np.repeat(distance_at_stim,init_frame), (num_stim, init_frame))
    cum_distance_post_stim=cum_travel_distance[:,init_frame:]-np.reshape(np.repeat(distance_at_stim,end_frame-init_frame), (num_stim, end_frame-init_frame))
    if cum_distance_prev_stim.shape[1]==0:
        total_distance_prev_stim=np.full((cum_distance_prev_stim.shape[0]), np.nan)
    else:
        total_distance_prev_stim=abs(cum_distance_prev_stim[:,0])
    if cum_distance_post_stim.shape[1]==0:
        total_distance_post_stim=np.full((cum_distance_post_stim.shape[0]), np.nan)
    else:
        total_distance_post_stim=cum_distance_post_stim[:,-1]

    distance_dict={"travel_distance_isi": total_distance_prev_stim,"travel_distance":total_distance_post_stim}
    summary_list.update(distance_dict)
    # convert the list into pandas dataframe
    summary_df = pd.DataFrame(summary_list) 
    summary_df["stim_type"] = stim_name_arr
    summary_df["present_trial_duration"] = present_duration_arr
    summary_df["preStim_ISI_duration"] = preStim_ISI_duration
    summary_df["stim_onset_thframe"]=stim_onset
    if "isi_onset" in locals():
        summary_df["isi_onset_thframe"]=isi_onset

    print("ready to plot graph with some data and decompose those lists")
    COL = MplColorHelper(colormap_name, 0, len(stim_type)-1)
    heading_angle_discretised_all=[]
    
    if visual_paradigm_name in ["gratings","sweeploom","conflict",'coherence',"densities","looming","combo"]:
        heading_angle=np.degrees(heading_angle)
    else:            
        fig = plt.figure(figsize=[24, 18])
        for i, this_stim_type in enumerate(stim_type):
            heading_angle_discretised=[]
            ax = fig.add_subplot(1,len(stim_type),i+1, polar=True)  # Create polar axes
            for this_index in list(np.where(stim_info['stim_type']==this_stim_type)[0]):
                step_index = diskretize(distance=cum_distance_post_stim[this_index,:],diskretise_length=BODY_LENGTH*num_step_per_decision)
                # angles = np.arctan2(np.diff(curated_Y), np.diff(curated_X)) ## in time series analysis, the angles should come from locustVR directly.
                # angles = np.insert(angles, 0, np.nan)
                these_heading_angles=heading_post_stim[this_index,step_index]
                if these_heading_angles.shape[0]>1:
                    these_heading_angles = (these_heading_angles + np.pi) % (2 * np.pi) - np.pi
                    #heading_angle_discretised.extend(np.degrees(these_heading_angles[1:]))
                    heading_angle_discretised.extend(these_heading_angles[1:])
            ax.hist(heading_angle_discretised,bins=12)
            ax.plot(circmean(heading_angle_discretised),7,marker='*',color='r',label='circular mean')
            #ax.set(title=title,ylim=[0,8],yticks=[4],yticklabels=[''])
            ax.set(title=this_stim_type,ylim=[0,8],yticks=[4],yticklabels=[''])
            #ax.set(title=this_stim_type,yticks=[1,3])
                # fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(27, 7), tight_layout=True)
                # ax1.scatter(np.arange(angles.shape[0]),angles)
                # ax2.scatter(np.arange(heading_angle.shape[0]),heading_angle)
                # ax3.scatter(np.arange(heading_angle[step_index].shape[0]),heading_angle[step_index])
                # ax1.set_title("discritised based on curated X and Y")
                # ax2.set_title("fictrac raw")
                # ax3.set_title("fictrac discritised based on new index")
                # fig.savefig(f"check_angle_{i}th_trial.png")
            heading_angle_discretised_all.append(heading_angle_discretised)
        #ax.legend()    
        fig.savefig(thisDir/"heading_distribution.png")
        fig.savefig(thisDir/"heading_distribution.svg")
        heading_angle=np.degrees(heading_angle)
    # shortened_isi=5
    # if shortened_isi>abs(analysis_window[0]):
    #     shortened_isi=abs(analysis_window[0])
    running_trial_distance_threshold=1*abs(analysis_window[0])#this threshold is defined based on travel distance 5-sec prior to stimulus onset     
    if analysis_methods.get("plotting_tbt_overview") == True:
   
        fig1 = plt.figure(figsize=(18, 7), tight_layout=True)
        ax = plt.subplot(241)
        if "opto_index" in summary_df.columns:
            ax.hist(summary_df["opto_index"].values, bins=20)
        # ax.set_xlabel("optomotor index")
        ax.set_ylabel("count")
        ax.set_title("after stimulus presentation")
        ax1 = plt.subplot(242)
        ax1.hist(summary_df["travel_distance"].values, bins=20)
        ax1.set_ylabel("count")
        ax1.set_title("after stimulus presentation")
        ax3 = plt.subplot(245)
        if "opto_index_isi" in summary_df.columns:
            ax3.hist(summary_df["opto_index_isi"].values, bins=20)
        ax3.set_xlabel("optomotor index")
        ax3.set_ylabel("count")
        ax3.set_title(f"{abs(analysis_window[0])} sec prior to the stimulus")
        ax4 = plt.subplot(246)
        ax4.hist(summary_df["travel_distance_isi"].values, bins=20)
        ax4.set_xlabel("travel distance (cm)")
        ax4.set_ylabel("count")
        ax4.set_title(f"{abs(analysis_window[0])} sec prior to the stimulus")
        
        ax_stationary = plt.subplot(243)
        ax_runnning = plt.subplot(247)
        ax_stationary.set_title(f"stationary trial (based on {abs(analysis_window[0])} sec prior to the stimulus)")
        ax_runnning.set_title(f"running trial (based on {abs(analysis_window[0])} sec prior to the stimulus)")
        for i in range(len(stim_type)):
            this_stim_type = stim_type[i]
            if type(stim_duration)==list:
                these_velocities = velocity_arr[(summary_df["stim_type"] == this_stim_type) & (summary_df['present_trial_duration']>=duration_for_optomotor_index),:]
                these_travel_distance_isi=summary_df["travel_distance_isi"][(summary_df["stim_type"] == this_stim_type) & (summary_df['present_trial_duration']>=duration_for_optomotor_index)].values
            else:
                these_velocities = velocity_arr[summary_df["stim_type"] == this_stim_type, :]
                these_travel_distance_isi=summary_df[summary_df["stim_type"] == this_stim_type]["travel_distance_isi"].values
            
            V_stationary=these_velocities[these_travel_distance_isi<running_trial_distance_threshold]
            time_points = np.arange(these_velocities.shape[1])
            if V_stationary.shape[0]>0:
                time_points_2D = np.tile(time_points, (V_stationary.shape[0], 1))
                ax_stationary.scatter(time_points_2D, V_stationary, s=1, color=COL.get_rgb(i),label=str(this_stim_type))
                #ax_stationary.plot(np.transpose(time_points_2D),np.transpose(V_stationary), color=COL.get_rgb(i),label=str(this_stim_type),linewidth=0.2,alpha=0.5)
            V_run=these_velocities[these_travel_distance_isi>running_trial_distance_threshold]
            if V_run.shape[0]>0:
                time_points_2D = np.tile(time_points, (V_run.shape[0], 1))
                ax_runnning.scatter(time_points_2D, V_run, s=1, color=COL.get_rgb(i),label=str(this_stim_type))
                #ax_runnning.plot(np.transpose(time_points_2D),np.transpose(V_run), color=COL.get_rgb(i),label=str(this_stim_type),linewidth=0.2,alpha=0.5)
        ax_runnning.legend()
        ax_stationary.legend()
        ax_runnning.set(
            ylim=[0, 25],
            ylabel="speed (cm/s)",
            xlabel="Time (frame no)",
        )
        ax_stationary.set(
            ylim=[0, 25],
            ylabel="speed (cm/s)",
        )
        ax_runnning.axvline(
            x=init_frame,
            color="black",
            linestyle="--",
            label="Stimulus Starts",
        )
        ax_stationary.axvline(
            x=init_frame,
            color="black",
            linestyle="--",
            label="Stimulus Starts",
        )
        ## draw subplot2 a heatmap of velocity
        ax2 = plt.subplot(2, 4, (4, 8))
        if analysis_methods.get("analysis_by_stimulus_type") == True:
            velocity_sorted_list = []
            for thisStim in stim_type:
                velocity_sorted_list.append(
                    velocity_arr[
                        np.where((summary_df.loc[:, "stim_type"] == thisStim)), :
                    ]
                )
            tmp = np.hstack(velocity_sorted_list)
            velocity_arr_for_colour_map = tmp[0, :, :]
            if create_null_null_condition:
                yticks = np.linspace(2, num_stim, num=len(stim_type),dtype=int)#np.arange(5,num_stim,len(stim_type)-1)
            else:
                yticks = np.arange(int(num_stim/len(stim_type)/2),num_stim,int(num_stim/len(stim_type)))
            if type(stim_type[0])==str:
                yticklabels = stim_type
            else:
                yticklabels = [str(this_stim_type) for this_stim_type in stim_type]
            # ax2.set(
            #     yticks=yticks,
            #     yticklabels=yticklabels,
            # )
            for this_y_mark in np.arange(int(num_stim/len(stim_type)),num_stim,int(num_stim/len(stim_type))):
                ax2.axhline(y=this_y_mark,xmin=0,xmax=velocity_arr_for_colour_map.shape[1],linewidth=0.3,color='w')
            #trans=mt.blended_transform_factory(ax2.transData,ax2.transAxes)
            #ax2.pylab.gca()
            #ax2.hlines(y=np.arange(int(num_stim/len(stim_type)),num_stim,int(num_stim/len(stim_type))),xmin=0,xmax=velocity_arr.shape[1],linewidth=10,color='w',transform=trans)
        else:
            ax2.set_yticks([])
            ax2.set_ylabel("Trial#")
            velocity_arr_for_colour_map = velocity_arr

        pcm = ax2.pcolormesh(velocity_arr_for_colour_map, cmap="magma", vmin=0, vmax=30)
        #pcm=ax2.pcolormesh(velocity_arr_for_colour_map, cmap = 'magma') ##sometimes, we dont need a vmax
        ## add colorbar
        fig1.colorbar(pcm)

        ## add vertical lines to indicate stimulus onset and offset
        ax2.axvline(
            x=init_frame,
            color="white",
            linestyle="--",
            label="Stimulus Onset",
        )
        if visual_paradigm_name == "sweeploom":
            approach_duration=np.unique(stim_info['Duration'].values)
            ax2.axvline(
            x=init_frame+presence_duration*monitor_fps,
            color="red",
            linestyle="--",
            label="Approach Onset",
        )
            ax2.axvline(
            x=init_frame+(presence_duration+approach_duration[0])*monitor_fps,
            color="red",
            linestyle="--",
            label="1st Approach Offset",
        )
            if approach_duration.shape[0]>1:
                ax2.axvline(
                x=init_frame+(presence_duration+approach_duration[1])*monitor_fps,
                color="red",
                linestyle="--",
                label="2nd Approach Offset",
            )

        if type(stim_duration) == list and align_with_isi_onset==False:
            pass
        elif align_with_isi_onset==True:
            pass
        else:
            ax2.axvline(
                x=end_frame,
                color="white",
                linestyle="--",
                label="Stimulus Ends",
            )
        if type(stim_duration) == list:
            frame_difference = (max(stim_duration) - min(stim_duration)) * camera_fps
            shortest_stim_end_frame = end_frame - frame_difference
            xticks = [0, init_frame, shortest_stim_end_frame, end_frame]
            xticklabels = [
                str(analysis_window[0]),
                "0",
                str(min(stim_duration)),
                str(max(stim_duration)),
            ]
        else:
            xticks = [init_frame, end_frame]
            xticklabels = ["0", str(stim_duration)]
        # ax2.set(
        #     xticks=xticks,
        #     xticklabels=xticklabels,
        #     xlabel="Time (s)",
        # )
        # ax2.set_xticks([init_frame])
        # ax2.set_xticklabels(["0"])
        ax2.set_xlabel("Time (s)")
        ## add title
        ax2.set_title("stimulus evoked velocity preview")
        if analysis_methods.get("analysis_by_stimulus_type") == True:
            plot_name = "optomotor_overview_by_type.png"
        else:
            plot_name = "optomotor_overview_by_time.png"
        if save_output == True:
            fig1.savefig(thisDir / plot_name)

    if visual_paradigm_name=='sweeping' or visual_paradigm_name=='looming':
        fig_stationary,fig_running,plot_tbt=plotting_visual_guided_response(velocity_arr,stim_type,summary_df,analysis_methods,running_trial_distance_threshold,COL)
        file_suffix = "_tbt" if plot_tbt else ""  
        if save_output == True:
            plot_name = f"visual_guided_velocity_different_duration_stationary{file_suffix}.png"
            fig_stationary.savefig(thisDir / plot_name)
            plot_name = f"visual_guided_velocity_different_duration_running{file_suffix}.png"
            fig_running.savefig(thisDir / plot_name)
        fig_stationary,fig_running,plot_tbt=plotting_visual_guided_response(heading_angle,stim_type,summary_df,analysis_methods,running_trial_distance_threshold,COL)
        file_suffix = "_tbt" if plot_tbt else "" 
        if save_output == True:
            plot_name = f"visual_guided_angular_velocity_{yaw_axis}axis_different_duration_stationary{file_suffix}.png"
            fig_stationary.savefig(thisDir / plot_name)
            plot_name = f"visual_guided_angular_velocit_{yaw_axis}axis_different_duration_running{file_suffix}.png"
            fig_running.savefig(thisDir / plot_name)

    if analysis_methods.get("plotting_position_dependant_fixation") == True:
        
        if 'LocustTexture1' in stim_info and 'R1' in stim_info:
            groups = stim_info.groupby(['R1', 'LocustTexture1'])
            summary_df=pd.concat([summary_df, stim_info['LocustTexture1'],stim_info['R1']], axis=1)
        else:
            groups = stim_info.groupby(['Duration','LocustTexture1']) 
        y_axis_lim=[-2.5,2.5]
        num_indep_variables=groups.ngroups
        colormap = np.array(["k", "r", "b", "g", "y", "m", "c", "k"])
        fig5, axes = plt.subplots(
            nrows=1, ncols=num_indep_variables, figsize=(18, 7), tight_layout=True
        )
        j=0
        for name, this_group in groups:
            for i in range(len(stim_type)):
                this_type_stim = stim_type[i]
                this_response=angular_velocity_tbt[np.where(this_group["stim_type"]==this_type_stim)[0],:]
                time_points = np.arange(this_response.shape[1])
                time_points_2D = np.tile(time_points, (this_response.shape[0], 1))
                if num_indep_variables==1:
                    axes.plot(
                        np.mean(time_points_2D, axis=0),
                        np.mean(this_response, axis=0),
                        color=colormap[i],
                        alpha=0.2,
                        label=this_type_stim,
                    )
                    axes.legend()
                else:
                    axes[j].plot(
                        np.mean(time_points_2D, axis=0),
                        np.mean(this_response, axis=0),
                        color=colormap[i],
                        alpha=0.2,
                        label=this_type_stim,
                    )
                    axes[j].legend()
            if num_indep_variables==1:
                axes.axvline(
                    x=init_frame,
                    color="black",
                    linestyle="--",
                    label="Stimulus Starts",
                )
                axes.set(title=f'locust texture{name[1]}, R{name[0]}',yticks=[y_axis_lim[0], 0, y_axis_lim[1]],ylim=y_axis_lim,ylabel="angular velocity (rad/s)")
            else:
                axes[j].axvline(
                    x=init_frame,
                    color="black",
                    linestyle="--",
                    label="Stimulus Starts",
                )
                axes[j].set(title=f'locust texture{name[1]}, R{name[0]}',yticks=[y_axis_lim[0], 0, y_axis_lim[1]],ylim=y_axis_lim,ylabel="angular velocity (rad/s)")
            j=j+1
        plot_name = f"conflict_analysis_overview.png"
        if save_output == True:
            fig5.savefig(thisDir / plot_name)

    if analysis_methods.get("plotting_optomotor_response") == True:
        # if 'cw' in stim_type[0]:
        #     colormap_name = "gnuplot"
        #     sm = cm.ScalarMappable(cmap=colormap_name)            
        #     COL = MplColorHelper(colormap_name, 0, len(stim_type)-1)
        target_speed=64*np.pi/180 #unit in radian/s
        fig2, (ax3,ax4,ax5,ax6) = plt.subplots(
            nrows=1, ncols=4, figsize=(72, 14), tight_layout=True
        )
        for i in range(len(stim_type)):
            this_stim_type = stim_type[i]
            if type(stim_duration)==list:
                this_response = summary_df.loc[(summary_df["stim_type"] == this_stim_type) & (summary_df['present_trial_duration']>=duration_for_optomotor_index),"opto_index"].values
            else:
                this_response = summary_df.loc[
                    summary_df["stim_type"] == this_stim_type, "opto_index"
                ].values
            if len(this_response) == 0:
                continue
            mean_response = np.nanmean(this_response, axis=0)
            sem_response = np.nanstd(this_response, axis=0, ddof=1) / np.sqrt(
                this_response.shape[0]
            )
            ax3.errorbar(
                this_stim_type,
                mean_response,
                yerr=sem_response,
                c="k",
                fmt="o",
                elinewidth=2,
                capsize=3,
            )
            if type(stim_duration)==list:
                this_heading_angle = heading_angle[(summary_df["stim_type"] == this_stim_type) & (summary_df['present_trial_duration']>=duration_for_optomotor_index),:]
                this_angular_velocity=angular_velocity_tbt[(summary_df["stim_type"] == this_stim_type) & (summary_df['present_trial_duration']>=duration_for_optomotor_index),:]
            else:
                this_heading_angle = heading_angle[summary_df["stim_type"] == this_stim_type, :]
                this_angular_velocity=angular_velocity_tbt[summary_df["stim_type"] == this_stim_type, :]
            time_points = np.arange(this_heading_angle.shape[1])
            time_points_2D = np.tile(time_points, (this_heading_angle.shape[0], 1))
            ax4.scatter(time_points_2D, this_heading_angle, s=0.5, color=COL.get_rgb(i),label=this_stim_type)
            #this_angular_velocity=np.diff(this_heading_angle)
            
            time_points = np.arange(this_angular_velocity.shape[1])
            time_points_2D = np.tile(time_points, (this_angular_velocity.shape[0], 1))
            ax5.plot(time_points, np.median(this_angular_velocity,axis=0),color=COL.get_rgb(i),label=this_stim_type)
            #this_angular_acceleration=np.diff(this_angular_velocity)
            this_angular_acceleration=np.gradient(this_angular_velocity, 1/camera_fps,axis=1)
            time_points = np.arange(this_angular_acceleration.shape[1])
            time_points_2D = np.tile(time_points, (this_angular_acceleration.shape[0], 1))
            ax6.plot(time_points, np.median(this_angular_acceleration,axis=0),color=COL.get_rgb(i),label=this_stim_type)
        
        dif_tmp= np.full(this_angular_velocity.shape[1],np.nan)
        stimulus_end_frame=(abs(analysis_window[0])+stim_duration)*camera_fps
        stim_w_array=np.ones(stim_duration*camera_fps)*target_speed
        dif_tmp[init_frame:stimulus_end_frame]=stim_w_array
        
        ax3.set_ylim(-1, 1)
        if visual_paradigm_name=='sweeping' or visual_paradigm_name=='gratings':
            ax3.tick_params("x", labelrotation=45)
            ax3.set_xticks(stim_type)
        ax3.set(
            xlabel=visual_paradigm_name,
            yticks=[-1, 0, 1],
            ylabel="Optomotor Index during stimulus"
        )
        # xticks=[np.max(stim_type), 0, np.min(stim_type)], np.sort(summary_df.stim_type.unique())
        ax4.axvline(
            x=init_frame,
            color="black",
            linestyle="--",
            label="Stimulus Starts",
        )
        ax4.axvline(
            x=stimulus_end_frame,
            color="black",
            linestyle="--",
            label="Stimulus Ends",
        )
        ax5.axvline(
            x=init_frame,
            color="black",
            linestyle="--",
            label="Stimulus Starts",
        )
        ax5.axvline(
            x=stimulus_end_frame,
            color="black",
            linestyle="--",
            label="Stimulus Ends",
        )
        ax6.axvline(
            x=init_frame,
            color="black",
            linestyle="--",
            label="Stimulus Starts",
        )
        ax6.axvline(
            x=stimulus_end_frame,
            color="black",
            linestyle="--",
            label="Stimulus Ends",
        )
        ax5.set_ylim(-4,4)
        ax6.set_ylim(-40, 40)
        ax5.plot(time_points,dif_tmp,color='k',linewidth=0.8)
        ax5.plot(time_points,dif_tmp*-1,color='k',linewidth=0.8)
        ax5.set(yticks=[-4, 0, 4])
        ax5.set_ylabel(r'Median angular velocity (rad s$^{-1}$)')
        ax6.set(yticks=[-50, 0, 50])
        ax6.set_ylabel(r'Median angular acceleration (rad s$^{-2}$)')
        yaw_orientation_limit = 180#only feasible if the animal turn a lot and the unit is in radian 
        y_axis_lim = [yaw_orientation_limit * -1, yaw_orientation_limit]
        # ax4.set_ylim(y_axis_lim[0], y_axis_lim[1])
        # if stim duration is a list, plot the max and min values
        if type(stim_duration) == list:
            frame_difference = (max(stim_duration) - min(stim_duration)) * camera_fps
            shortest_stim_end_frame = end_frame - frame_difference
            xticks = [0, init_frame, shortest_stim_end_frame, end_frame]
            xticklabels = [
                str(analysis_window[0]),
                "0",
                str(min(stim_duration)),
                str(max(stim_duration)),
            ]
        else:
            xticks = [0, init_frame, end_frame]
            xticklabels = [str(analysis_window[0]), "0", str(stim_duration)]
        ax4.set(
            #yticks=[y_axis_lim[0], 0, y_axis_lim[1]],
            #ylim=[y_axis_lim[0], y_axis_lim[1]],
            ylabel="Orientation (degree)",
            xticks=xticks,
            xticklabels=xticklabels,
            xlabel="Time (s)",
        )
        tick_positions = np.linspace(0, 1, len(stim_type))
        tick_labels = [str(value) for value in stim_type]
        cbar = plt.colorbar(sm, ax=ax4, orientation="vertical", ticks=tick_positions)
        cbar.set_ticklabels(tick_labels)

        plot_name = f"optomotor_analysis_until{duration_for_optomotor_index}s.png"
        if save_output == True:
            fig2.savefig(thisDir / plot_name)

    if analysis_methods.get("plotting_deceleration_accerleration") == True:
        (
            _,
            walk_states_tbt,
            turn_states_tbt,
            _,
            _,
        ) = analysis_stim_evoked_response_combo(
            stim_to_analyse,
            stim_info,
            walk_states,
            turn_states,
            tem_df_this_exp,
            analysis_methods
        )
        walk_states_tbt=walk_states_tbt.astype(bool)
        turn_states_tbt=turn_states_tbt.astype(bool)
        #or use velocity_arr
        stop_index = detect_stop_events(arr=velocity_arr, analysis_methods=analysis_methods,response_window=[-0.5,4.5],sustained_time_for_behaviour=0.5,save_path=thisDir)
        summary_df['stop_index']=stop_index
        fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(36, 14), tight_layout=True)
        i=0
        for name, group in summary_df.groupby('stim_type'):
            if analysis_methods.get("active_trials_only")==True:
                trial_to_analyse=stop_index[group.index][group['travel_distance_isi']>running_trial_distance_threshold]
            else:
                trial_to_analyse=stop_index[group.index]
            stop_to_analyse=trial_to_analyse[trial_to_analyse>init_frame]
            if trial_to_analyse.shape[0]>0:
                stop_ratio=stop_to_analyse.shape[0]/trial_to_analyse.shape[0]
                print(f'stop/deceleration events for stimulus {name} is {stop_ratio}')
                ax5.scatter(np.ones(stop_to_analyse.shape[0])*i,stop_to_analyse/camera_fps)
            i=i+1
        plot_name = f"stop_distribution.png"
        if save_output == True:
            fig5.savefig(thisDir / plot_name)      
        # dev_trial = trial_to_analyse[dev_trial_id, :]
        # ##When analysing recover epochs, focus on the trials has similar deceleration dynamics
        # index_list = recovery_detection(dev_trial, dev_timing_from_analysis_window)
        # rec_array = (
        #     index_list - dev_timing_from_analysis_window
        # )  # this normalise the recovery duration from the first deceleration

    if save_output == True and overwrite_curated_dataset == True:
        parquet_directory = thisDir / "behavioural_summary.parquet.gzip"
        summary_df.to_parquet(parquet_directory,compression='gzip')
        data_dict = {name: col.values for name, col in summary_df.items()}
        summary_file_name = thisDir / "behavioural_summary.mat"
        savemat(summary_file_name, data_dict)
        velocity_file_name = thisDir / "velocity_tbt.npy"
        with open(velocity_file_name, "wb") as f:
            np.save(f, velocity_arr)
        file_name = thisDir / "angular_velocity_tbt.npy"
        with open(file_name, "wb") as f:
            np.save(f, angular_velocity_tbt)
        print("Save velocity, angular velocity into array and behavioural summary as dataframe. Move to the next animal")
        json_string = json.dumps(analysis_methods, indent=1)
        with open(thisDir / "analysis_methods_dictionary_backup.json", "w") as f:
            f.write(json_string)

    elif overwrite_curated_dataset == False:
        print(f"dont overwrite existing database.")
    else:
        print("dont save anything")

    return summary_df, velocity_arr, angular_velocity_tbt, travel_distance_whole_session,heading_angle_discretised_all,walk_states,turn_states,turn_cw,turn_ccw


if __name__ == "__main__":
    # weird files 50525355565759606269727679 fixed by threshold 46 47 52 59 53 fixed by filtering trial in different way 4150556272
    # GN24098 seems not follow the dot motion
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24021/240717/coherence/session1"  # good for checking fft, where potential false nagative show up;1:11:33;1:15:08
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24078/240804/coherence/session1"  # good for checking fft, where 2 comes from
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24076/240804/coherence/session1"
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24084/240805/coherence/session1"# good for checking fft, where false nagative show up
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24114\240811\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24098\240810\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24089\240810\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24036\240801\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24133\241210\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24126\241209\coherence\session1"
    # thisDir = "Z:\\DATA\\experiment_trackball_Optomotor\\MatrexVR\\GN24133\\241210\\coherence\\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24136\241210\coherence\session1"
    #thisDir = r"Z:\\DATA\\experiment_trackball_Optomotor\\MatrexVR\\GN24124\\241209\\coherence\\session1"
    #thisDir = r"C:\Users\neuroLaptop\Documents\GN25006\250312\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25009\250403\coherence\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25007\250319\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\VCCball\L23016\230328\densities"
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/VCCball/L22056/220905/densities"
    #thisDir = r"C:\Users\neuroLaptop\Documents\GN25040\250106\conflict\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN25026\250106\conflict\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN25040\250106\conflict\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25102\250929\looming\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25101\250917\looming\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25102\250929\coherence\session1"
    #thisDir = r"Y:\GN25051\251101\looming\session2"# locust loom prestim last 60 sec-only #non-walking session
    #thisDir = r"Y:\GN25053\251108\looming\session2"# locust loom prestim last 60 sec-only #non-walking session
    #thisDir = r"Y:\GN25053\251108\looming\session6"# locust loom prestim last 60 sec-only walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25058\251123\looming\session3"#locust loom
    #thisDir = r"Y:\GN25060\251130\looming\session1"#locust loom
    #thisDir = r"Y:\GN25061\251205\looming\session3"#locust loom
    #thisDir = r"Y:\GN25053\251108\looming\session5"# sequential looming, so the prestim last 60 sec walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25060\251130\looming\session2"# sequential looming, so the prestim last 60 sec walking session, did not walk much
    #thisDir = r"Y:\GN25062\251206\looming\session2"# sequential looming, walking session
    #thisDir = r"Y:\GN25061\251205\looming\session2"# sequential looming
    #thisDir = r"Y:\GN25061\251205\looming\session4"
    #thisDir = r"Y:\GN25060\251130\looming\session4"#non-walking session
    #thisDir = r"Y:\GN25059\251129\looming\session1"#walking session #angular velocity looks weird
    #thisDir = r"Y:\GN25058\251123\looming\session2"
    #thisDir = r"Y:\GN25057\251122\looming\session2"
    #thisDir = r"Y:\GN25053\251108\looming\session7"# walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25053\251108\looming\session1"# walking session, just one or two jumping epochs
    #thisDir = r"Y:\GN25052\251102\looming\session1"# walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25051\251101\looming\session1"# walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25062\251206\sweeping\session1"# bilateral sweeping, there were problem with stimulus combination. NB and BN were not used. Instead, additional NW (0,0,0,0,1,1,1,1) and WN (1,1,1,1,0,0,0,0)
    #thisDir = r"Y:\GN25061\251205\sweeping\session1"
    #thisDir = r"Y:\GN25060\251130\sweeping\session1"
    #thisDir = r"Y:\GN25059\251129\sweeping\session1"
    #thisDir = r"Y:\GN25057\251122\sweeping\session1"
    #thisDir = r"Y:\GN25052\251102\sweeping\session1"
    #thisDir = r"Y:\GN25051\251101\sweeping\session1"
    #thisDir = r"Y:\GN25053\251108\coherence\session1"##had issue detecting transition onset. prestim lasts 180 sec only. Animal did not move much.
    #thisDir = r"Y:\GN25054\251109\coherence\session1"#not moving at all
    #thisDir = r"Y:\GN25055\251115\coherence\session1"# walking session
    #thisDir = r"Y:\GN25055\251115\coherence\session2"# walking session
    #thisDir = r"Y:\GN25056\251116\coherence\session1"# walking session
    #thisDir = r"Y:\GN25056\251116\coherence\session2"# walking session, either animal jump or tracking error
    #thisDir = r"Y:\GN25057\251122\coherence\session1"# walking session
    #thisDir = r"Y:\GN25051\251101\coherence\session1"
    #thisDir = r"Y:\GN25060\251130\coherence\session2"#vertical coherence
    #thisDir = r"Y:\GN25059\251129\coherence\session2"#vertical coherence
    #thisDir = r"Y:\GN25059\251129\coherence\session1"#horizontal coherence
    #thisDir = r"Y:\GN25062\251206\coherence\session2"#horizontal coherence, not walking session
    #thisDir = r"Y:\GN25062\251206\coherence\session1"#vertical coherence, not walking session
    #thisDir = r"Y:\GN25060\251130\coherence\session1"#horizontal coherence, can be a good example
    #thisDir = r"Y:\GN25052\251102\coherence\session1"#had issue detecting transition onset. prestim lasts 180 sec only. Animal did not move much.
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25103\251222\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25102\250929\sweeping\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25103\251222\sweeping\session2"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25103\251222\looming\session1"
    #thisDir = r"C:\Users\neuroLaptop\Documents\GN25060\251130\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25106\251229\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25105\251229\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25104\251223\coherence\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25104\251223\conflict\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25104\251223\sweeping\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25105\251229\sweeping\session1"## needs a new scene name for bilateral ccw cw sequence
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25106\251229\sweeping\session1"## needs a new scene name for bilateral ccw cw sequence
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26101\260107\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26101\260108\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26102\260112\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26103\260112\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26103\260112\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26102\260112\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26104\260119\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26104\260120\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26104\260119\coherence\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26104\260120\sweeping\session2"# the isi in bilateral non-rotation sequence is 10 or 20 because too many conditions
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26104\260120\conflict\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26104\260119\sweeping\session2"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26105\260119\sweeping\session2"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26105\260120\sweeping\session2"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26105\260119\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26106\260126\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26107\260126\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26109\260127\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26106\260126\sweeping\session2"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26106\260126\conflict\session1"
    #thisDir =r"Y:\GN26008\250727\gratings\session1"
    #thisDir =r"Y:\GN26013\260214\sweeping\session1"
    #thisDir =r"Y:\GN26014\260215\sweeping\session1"
    #thisDir =r"Y:\GN26012\260208\sweeping\session1"
    #thisDir =r"Y:\GN26011\260207\sweeping\session2"
    #thisDir =r"Y:\GN26010\260201\sweeping\session2"
    #thisDir =r"Y:\GN26007\260124\sweeping\session1"
    #thisDir =r"Y:\GN26008\260125\sweeping\session1"
    #thisDir =r"Y:\GN26009\260131\sweeping\session2"
    #thisDir =r"Y:\GN25062\251206\looming\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26125\260223\sweeping\session2"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26119\260216\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26126\260223\choices\session1"
    #thisDir =r"Y:\GN26016\260221\choices\session1"
    #thisDir =r"Y:\GN26017\260222\choices\session1"
    #thisDir =r"Y:\GN26015\260221\choices\session1"
    #thisDir =r"Y:\GN26018\260228\choices\session2"
    #thisDir =r"Y:\GN26017\260222\choices\session1"
    #thisDir =r"Y:\GN26090\260801\looming\session1"
    #thisDir =r"Y:\GN26040\260411\looming\session1"
    #thisDir =r"Y:\GN26042\260412\looming\session2"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\VCCball\L22117\221015\sweeploom"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\VCCball\L23063\230831\sweeploom"
    #thisDir =r"Y:\GN26056\260509\looming\session1"
    #thisDir ="/Users/chiyu_lee/OpenEphys/GN26063/260523/sweeping/session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26192\260718\looming\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26187\260613\looming\session1"
    #thisDir =r"Y:\GN26063\260523\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26186\260604\looming\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26134\260309\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26105\260119\coherence\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26105\260119\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26105\260120\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26106\260126\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26107\260126\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26108\260127\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26109\260127\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26114\260209\sweeping\session2"# tracking after 30000 frames has problem, increased threshold improved but can be better
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26113\260203\sweeping\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26115\260209\choices\session1"
    #thisDir =r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN26118\260210\choices\session1"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
