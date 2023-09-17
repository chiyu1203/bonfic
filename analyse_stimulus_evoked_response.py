"""spontaneous response: needs to figure out stationary, moving, timing and total travel distance as a overview 
event evoked response: needs to divide stationary trails, moving trails 
(in the case of optomotor response, need to subdivide them into moving along stimulus, 
moving against stimulus and others)
event evoked response: need to output the trial id (both within the same stimulus type and overall) and response, 
velocity vector across time in a trial plus optomotor response
analyis methods needs to tell us what to analye. For example optomotor response, acceleration epochs, deceleration epochs, transition between stationary to walk and walk to stationary
"""
import json
import time
import os, fnmatch
import pandas as pd
import numpy as np
import warnings
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

csv_pattern = "trial_*.csv"
database_pattern = "*.pickle"
exp_name = "dots"
frame_rate = 144
stim_duration = 25
grey_duration = 5
# analysis_window=frame_rate*(stim_duration+grey_duration)
analysis_window = [grey_duration * -1, stim_duration]
running_speed_threshold = 50
turning_threshold = 0.33
window_length = analysis_window[1] - analysis_window[0]
# stim_type=["clock_coh100","clock_coh75","clock_coh50","clock_coh25","clock_coh0","anticlock_coh100","anticlock_coh75","anticlock_coh50","anticlock_coh25"]
stim_type = [
    100,
    75,
    50,
    25,
    0,
    -25,
    -50,
    -75,
    -100,
]  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve


def optomotor_response_idx(ev_dat):
    opt_idx = (
        np.nansum(abs(ev_dat[ev_dat < 0])) - np.nansum(abs(ev_dat[ev_dat > 0]))
    ) / (np.nansum(abs(ev_dat[ev_dat < 0])) + np.nansum(abs(ev_dat[ev_dat > 0])))
    return opt_idx


def main(thisDir, json_file):
    with open(json_file, "r") as f:
        analysis_methods = json.loads(f.read())
    file = os.listdir(thisDir)
    file_name = fnmatch.filter(file, database_pattern)
    database = os.path.join(thisDir, file_name[0])
    df = pd.read_pickle(database)
    exp_date = thisDir.split("/")[5]
    if analysis_methods.get("filtering_method") == "sg_filter":
        x_all = savgol_filter(df.loc[:, ["intergrated x position"]], 71, 3, axis=0)
        y_all = savgol_filter(df.loc[:, ["intergrated y position"]], 71, 3, axis=0)
        instant_v = savgol_filter(df.loc[:, ["instant speed"]], 39, 3, axis=0)
        runmean_v = df["instant speed"].rolling(10).mean().values
        # vx=savgol_filter(df.loc[:, ['instant speed']], 39, 3, axis=0)
        # vz=savgol_filter(df.loc[:, ['instant speed']], 39, 3, axis=0)
    else:
        x_all = df.loc[:, ["intergrated x position"]]
        y_all = df.loc[:, ["intergrated y position"]]
    ## here write an additional function to classify stationary, moving and total travel distance
    if analysis_methods.get("plotting_trajectory") == True:
        plt.plot(x_all, y_all, c=np.arange(len(y_all)), marker=".", alpha=0.5)
        plt.show()

    ##here to start analying stimulus-evoked response.
    ##Firstly, load stimulus information
    csv_file = fnmatch.filter(file, csv_pattern)
    csv_file_directory = os.path.join(thisDir, csv_file[0])
    stim_info = pd.read_csv(csv_file_directory)
    stim_info = stim_info.reset_index()
    stim_variable = []
    stimulus_timestamp = []
    if analysis_methods.get("experiment_name") == "gratings":
        for row in range(0, len(stim_info)):
            # delete the bracket in current speed and save it as integer in new list
            stim_variable.append(int(stim_info.loc[row, "Frame"].replace("(", "")))
            # combine the last two values into the timestamp (were divided into 2 columns cause german computer
            # used comma as decimal seperator and save timestamp as float in new file
            if isinstance(stim_info.loc[row, "Value"], float):
                timestamp = stim_info.loc[row, "Timestamp"]
            else:
                timestamp = (
                    str(stim_info.loc[row, "Timestamp"])
                    + "."
                    + stim_info.loc[row, "Value"]
                )
            stimulus_timestamp.append(float(timestamp.replace(")", "")))
        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))

    elif analysis_methods.get("experiment_name") == "sweeploom":
        for row in range(0, len(stim_info)):
            stim_variable.append(int(stim_info.iloc[row, 2].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 3].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 4].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 5].split("=", 1)[1]))
            # stim_variable.append(int(re.split(r'=|}',stim_info.iloc[row,5])[1]))
            if isinstance(stim_info.loc[row, "Value"], float):
                timestamp = stim_info.loc[row, "Timestamp"]
            else:
                timestamp = (
                    str(stim_info.loc[row, "Timestamp"])
                    + "."
                    + stim_info.loc[row, "Value"]
                )
            stimulus_timestamp.append(float(timestamp.replace(")", "")))
        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))
        # some bugs in the bonsai workflow. Needs to remove the first timestamp and the last row of stim variable here
        stimulus_timestamp.pop(0)
        stim_variable = np.delete(stim_variable, -1, axis=0)
        stim_df = pd.DataFrame(stim_variable)
        if exp_date in ["221015", "221016"]:
            filters = [
                (stim_df.iloc[:, 0] == 1) & (stim_df.iloc[:, 2] > stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 0] == 8) & (stim_df.iloc[:, 2] > stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 0] == 1) & (stim_df.iloc[:, 2] < stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 0] == 8) & (stim_df.iloc[:, 2] < stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 0] == 1) & (stim_df.iloc[:, 2] == stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 0] == 8) & (stim_df.iloc[:, 2] == stim_df.iloc[:, 3]),
            ]

        else:
            filters = [
                (stim_df.iloc[:, 6] == 1) & (stim_df.iloc[:, 2] > stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 6] == 2) & (stim_df.iloc[:, 2] > stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 6] == 1) & (stim_df.iloc[:, 2] < stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 6] == 2) & (stim_df.iloc[:, 2] < stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 6] == 1) & (stim_df.iloc[:, 2] == stim_df.iloc[:, 3]),
                (stim_df.iloc[:, 6] == 2) & (stim_df.iloc[:, 2] == stim_df.iloc[:, 3]),
            ]
        ### use this criteria for experiments during exp_date=['221015','221016']

    elif analysis_methods.get("experiment_name") == "dots":
        for row in range(0, len(stim_info)):
            stim_variable.append(int(stim_info.iloc[row, 2].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 4].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 5].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 7].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 11].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 13].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 17].split("=", 1)[1]))
            if isinstance(stim_info.loc[row, "Value"], float):
                timestamp = stim_info.loc[row, "Timestamp"]
            else:
                timestamp = (
                    str(stim_info.loc[row, "Timestamp"])
                    + "."
                    + stim_info.loc[row, "Value"]
                )
            stimulus_timestamp.append(float(timestamp.replace(")", "")))
        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))
        stim_df = pd.DataFrame(stim_variable)
        filters = [
            (stim_df.iloc[:, 5] < 1) & (stim_df.iloc[:, 6] == 100),
            (stim_df.iloc[:, 5] < 1) & (stim_df.iloc[:, 6] == 75),
            (stim_df.iloc[:, 5] < 1) & (stim_df.iloc[:, 6] == 50),
            (stim_df.iloc[:, 5] < 1) & (stim_df.iloc[:, 6] == 25),
            (stim_df.iloc[:, 5] < 1) & (stim_df.iloc[:, 6] == 0),
            (stim_df.iloc[:, 5] > 1) & (stim_df.iloc[:, 6] == 25),
            (stim_df.iloc[:, 5] > 1) & (stim_df.iloc[:, 6] == 50),
            (stim_df.iloc[:, 5] > 1) & (stim_df.iloc[:, 6] == 75),
            (stim_df.iloc[:, 5] > 1) & (stim_df.iloc[:, 6] == 100),
        ]
    else:
        warnings.warn("stimulus paradigm can not be recognised")
        return

    # compare the timestamp between event file and dat file
    data_timestamp = df.loc[:, "timestamp"]
    stimulus_timestamp_b = np.tile(stimulus_timestamp, (len(data_timestamp), 1))
    data_timestamp_b = np.tile(data_timestamp, (len(stimulus_timestamp), 1))
    timestamp_diff = data_timestamp_b > np.transpose(stimulus_timestamp_b)
    stim_onset = np.where(np.roll(timestamp_diff, 1, axis=1) != timestamp_diff)[1][1::2]
    stim_df["stim_onset"] = stim_onset
    stim_df["stim_type"] = np.select(filters, stim_type)
    # travel_distance_all_trials=[]
    # travel_of_interest_in_the_experiment=[]
    # opto_of_interest_in_the_experiment=[]
    travel_distance_during_stim_all = []
    travel_distance_prior_stim_all = []
    optomotor_response_during_stim_all = []
    velocity_list = []
    index_list = []
    opto_running_trials = []
    travel_distance_running_trials = []
    x_axis_value_running_trials = []
    init_time = abs(analysis_window[0])
    init_frame = init_time * frame_rate
    end_time = abs(analysis_window[0]) + stim_duration
    end_frame = end_time * frame_rate
    vz = df.loc[:, ["delta rotation vector lab z"]].values
    if analysis_methods.get("analyse_stimulus_by_time") == True:
        stim_to_analyse = stim_df.loc[:, "stim_onset"]
        stim_name_list = np.array(stim_df["stim_type"])
        grey_onset = (
            stim_to_analyse
            + np.ones(len(stim_to_analyse), dtype=int) * analysis_window[0] * frame_rate
        )
        analysie_window_tile = np.tile(
            np.arange(window_length * frame_rate), (len(grey_onset), 1)
        )
        index_points = (
            np.array(grey_onset).reshape(len(grey_onset), 1) + analysie_window_tile
        )
        velocity_z = vz[index_points]
        opto_all_trials = np.ones(velocity_z.shape[0])
        for i in range(0, velocity_z.shape[0]):
            opto_all_trials[i] = optomotor_response_idx(velocity_z[i, init_frame:, 0])

        x = x_all[index_points]
        y = y_all[index_points]
        x_dif = np.diff(x, axis=1)
        y_dif = np.diff(y, axis=1)
        travel_distance_fbf = np.sqrt(np.add(np.square(x_dif), np.square(y_dif)))
        velocity = travel_distance_fbf * frame_rate
        # trial_num.append(velocity.shape[0])
        velocity_list.append(velocity[:, :, 0])
        cum_travel_distance = np.cumsum(
            np.sqrt(np.add(np.square(x_dif), np.square(y_dif))), axis=1
        )
        travel_distance_prior_stim = cum_travel_distance[:, init_frame, 0]
        travel_distance_during_stim = (
            cum_travel_distance[:, -1, 0] - travel_distance_prior_stim
        )

        summary_list = {
            "opto_index": opto_all_trials,
            "travel_distance": travel_distance_during_stim,
            "optoxcoherence": opto_all_trials * stim_name_list,
            "stim_index": stim_to_analyse.index,
            "coherence_level": stim_name_list,
        }
        summary_df = pd.DataFrame(summary_list)
        # summary_df.loc[summary_df["coherence_level"]==100]
        # summary_df.set_index(["stim_index", "coherence_level"], inplace=True)
        ## classify optomotor response
        turning_trials = summary_df.loc[
            abs(summary_df["opto_index"]) > turning_threshold
        ]
        running_trials = summary_df.loc[
            summary_df["travel_distance"] > running_speed_threshold
        ]
        turning_walk_trials = summary_df.loc[
            (summary_df["travel_distance"] > running_speed_threshold)
            & (abs(summary_df["opto_index"]) > turning_threshold)
        ]
        straight_walk_trials = summary_df.loc[
            (summary_df["travel_distance"] > running_speed_threshold)
            & (abs(summary_df["opto_index"]) < turning_threshold)
        ]
        follow_trials = turning_walk_trials.loc[
            turning_walk_trials["optoxcoherence"] < 0
        ]
        against_trials = turning_walk_trials.loc[
            turning_walk_trials["optoxcoherence"] > 0
        ]
        spontaneous_trials = turning_walk_trials.loc[
            turning_walk_trials["optoxcoherence"] == 0
        ]
        tmp = follow_trials.groupby("coherence_level").count()
        follow_count_coherence = tmp.index.values
        follow_count = tmp.values[:, 0]

        ##traditional way that does not use pandas
        """
        turning_trials = abs(opto_all_trials) > turning_threshold
        running_trial = travel_distance_during_stim > running_speed_threshold
        turning_walk_trials = opto_all_trials[
            (turning_trials == True) & (running_trial == True)
        ]
        stim_coherence_interest = stim_name_list[
            (turning_trials == True) & (running_trial == True)
        ]
        # opto_interest=opto_all_trials[turning_walk_trials]
        straight_walk_trials = opto_all_trials[
            (turning_trials == False) & (running_trial == True)
        ]
        stationary_trials = opto_all_trials[running_trial == False]
        follow_trials = turning_walk_trials * stim_coherence_interest < 0
        against_trials = turning_walk_trials * stim_coherence_interest > 0
        spontaneous_turn_trials = turning_walk_trials * stim_coherence_interest == 0
        follow_trials_coherence = stim_coherence_interest[follow_trials]
        against_trials_coherence = stim_coherence_interest[against_trials]
        follow_trials_response = turning_walk_trials[follow_trials]
        against_trials_response = turning_walk_trials[against_trials]
        spontaneous_trials_response = turning_walk_trials[spontaneous_turn_trials]
        """
        travel_distance_during_stim_all.append(travel_distance_during_stim)
        travel_distance_prior_stim_all.append(travel_distance_prior_stim)
        optomotor_response_during_stim_all.append(opto_all_trials)
        # travel_distance_tbt=np.concatenate(travel_distance_prior_stim_all)
        ##concatenate travel distance for histogram

        ##instead of stim_duration, I used a fixed 3 sec that is the shortest stim_duration here
        print("ready to plot graph with some data")
        # if analysis_methods.get("plotting_event_related_histogram")== True:
        #     fig, axes = plt.subplots(1,len(opto_running_trials),figsize=(40,3),squeeze=False)
        #     for i, ax in zip(range(0,index_points.shape[1]),axes.reshape(-1)):
        #         this_response=opto_running_trials[i]
        #         this_coherence=x_axis_value_running_trials[i]
        #         ax.hist(this_response)
        #         ax.set_title(f'stimulus coherence level: {this_coherence}')
        #     plt.show()

        if analysis_methods.get("plotting_tbt_overview") == True:
            optomotor_response_during_stim = np.concatenate(
                optomotor_response_during_stim_all
            )
            travel_distance_prior_stim = np.concatenate(travel_distance_prior_stim_all)
            travel_distance_during_stim = np.concatenate(
                travel_distance_during_stim_all
            )

            ##concatenate velocity list for heatmap
            velocity_list = np.vstack(velocity_list)
            ##arrange xticks at onset and offset of the stimulus for heatmap
            heatmap_xticks = [init_frame, end_frame]
            # trial_num_across_stim=np.cumsum(trial_num)
            fig1, (ax, ax1, ax2) = plt.subplots(
                nrows=1, ncols=3, figsize=(18, 7), tight_layout=True
            )
            ax.hist(optomotor_response_during_stim, bins=20)
            ax.set_xlabel("optomotor index")
            ax.set_ylabel("frequency")
            ax.set_title("optomotor response during stimulus presentation")
            ax1.hist(travel_distance_during_stim, bins=20)
            ax1.set_xlabel("travel distance (mm)")
            ax1.set_ylabel("frequency")
            ax1.set_title("travel distance during stimulus presentation")
            ## draw subplot2
            pcm = ax2.pcolormesh(velocity_list, cmap="magma", vmin=0, vmax=100)
            # pcm=ax2.pcolormesh(velocity_tbt, cmap = 'magma') ##sometimes, we dont need a vmax
            ## add colorbar
            fig1.colorbar(pcm)
            ## add horizontal lines to indicate each stimulus type
            # for ind in range(0,trial_num_across_stim.shape[0]):
            #     ax2.axhline(y=trial_num_across_stim[ind], color='white', linestyle='-', label='Stimulus Type',linewidth=0.5)
            ## remove y ticks for clearity and add y label
            ax2.set_yticks([])
            ax2.set_ylabel("Trial#")
            ## add vertical lines to indicate stimulus onset and offset
            ax2.axvline(
                x=heatmap_xticks[0],
                color="white",
                linestyle="-",
                label="Stimulus Starts",
            )
            # ax2.axvline(x=heatmap_xticks[1], color='white', linestyle='--', label='Stimulus Ends')
            # ax2.axvline(x=heatmap_xticks[1]+1*frame_rate, color='white', linestyle='--', label='Slow Stimulus Ends')
            # ax2.set_xticks([heatmap_xticks[0],heatmap_xticks[1]])
            # ax2.set_xticklabels(['0','3'])
            ax2.set_xticks([heatmap_xticks[0]])
            ax2.set_xticklabels(["0"])
            ax2.set_xlabel("Time (s)")
            ## add title
            ax2.set_title("stimulus evoked velocity preview")
            plot_name = "optomotor_overview.png"
            fig1.savefig(os.path.join(thisDir, plot_name))

        if analysis_methods.get("plotting_optomotor_response") == True:
            colormap = np.array(["r", "y", "m", "c", "k", "b", "g", "r"])
            fig2, (ax3, ax4) = plt.subplots(
                nrows=1, ncols=2, figsize=(18, 7), tight_layout=True
            )
            for i in range(len(stim_type)):
                this_coherence = stim_type[i]
                this_response = summary_df.loc[
                    summary_df["coherence_level"] == this_coherence, "opto_index"
                ].values
                # this_coherence = x_axis_value_running_trials[i]
                mean_response = np.mean(this_response, axis=0)
                sem_response = np.std(this_response, axis=0, ddof=1) / np.sqrt(
                    this_response.shape[0]
                )
                ax3.errorbar(
                    this_coherence,
                    mean_response,
                    yerr=sem_response,
                    c=colormap[0],
                    fmt="o",
                    elinewidth=2,
                    capsize=3,
                )
            ax3.set_ylim(-1, 1)
            # ax3.set_yticks([-1, 0, 1])
            ax3.set(
                yticks=[-1, 0, 1],
                ylabel="Optomotor Index",
                xticks=[100, 50, 0, -50, -100],
                xlabel="Coherence level (%)",
            )
            # ax3.set_yticklabels(["-1", "0", "1"], size=16)
            # ax3.set_xticks([100, 50, 0, -50, -100])
            # ax.set_xticklabels(['100','75','50','25','0','25','50','75','100'],size=16)
            # ax3.set_xlabel("Coherence level (%)", fontsize=18)
            # ax3.set_ylabel("Optomotor Index", fontsize=18)
            ax4.scatter(follow_count_coherence, follow_count, c=colormap[0], marker="o")
            ax4.set_ylim(0, 15)
            ax4.set(
                yticks=[0, 15],
                ylabel="Follow response (count)",
                xticks=[100, 50, 0, -50, -100],
                xlabel="Coherence level (%)",
            )
            # ax4.set_ylim(0, 15)
            # ax4.set_yticks([0, 15])
            # ax.set_xticks([1,2,3,4,5,6,7,8,9])
            # ax.set_xticklabels(['100','75','50','25','0','25','50','75','100'],size=16)
            # ax4.set_xlabel("Coherence level (%)", fontsize=18)
            # ax4.set_ylabel("Follow response (count)", fontsize=18)
            plot_name = "optomotor_analysis.svg"
            fig2.savefig(os.path.join(thisDir, plot_name))

        # print(len(velocity_tbt))
        # print(len(travel_distance_during_stim_all))
        # try not to return variables as list. Instead, outputing them as array
        plt.show()
        return summary_df, velocity_list, travel_distance_during_stim_all

    else:
        for stim_types in range(0, len(stim_type)):
            this_stim = stim_type[stim_types]
            stim_to_analyse = stim_df.loc[
                np.where((stim_df.loc[:, "stim_type"] == this_stim)), "stim_onset"
            ]
            stim_name_list = stim_df.loc[stim_to_analyse.index, "stim_type"].values
            grey_onset = (
                stim_to_analyse
                + np.ones(len(stim_to_analyse), dtype=int)
                * analysis_window[0]
                * frame_rate
            )
            analysie_window_tile = np.tile(
                np.arange(window_length * frame_rate), (len(grey_onset), 1)
            )
            index_points = (
                np.array(grey_onset).reshape(len(grey_onset), 1) + analysie_window_tile
            )

            velocity_z = vz[index_points]
            velocity_list.append(velocity_z[:, :, 0])
            opto_all_trials = []
            for i in range(0, velocity_z.shape[0]):
                test = optomotor_response_idx(
                    velocity_z[i, init_time * frame_rate :, 0]
                )
                opto_all_trials.append(test)

            x = x_all[index_points]
            y = y_all[index_points]
            x_dif = np.diff(x, axis=1)
            y_dif = np.diff(y, axis=1)
            travel_distance_fbf = np.sqrt(np.add(np.square(x_dif), np.square(y_dif)))
            cum_travel_distance = np.cumsum(
                np.sqrt(np.add(np.square(x_dif), np.square(y_dif))), axis=1
            )
            travel_distance_prior_stim = cum_travel_distance[
                :, init_time * frame_rate, 0
            ]
            travel_distance_during_stim = (
                cum_travel_distance[:, -1, 0] - travel_distance_prior_stim
            )
            travel_distance_during_stim_all.append(travel_distance_during_stim)
            index_list.append(stim_to_analyse.index)

            ## analyse focus on moving trials
            trial_of_interest = travel_distance_during_stim > running_speed_threshold
            opto_of_interest = np.array(opto_all_trials)[trial_of_interest]
            travel_of_interest = travel_distance_during_stim[trial_of_interest]
            opto_running_trials.append(opto_of_interest)
            travel_distance_running_trials.append(travel_of_interest)
            # coherence_value=np.ones(trial_of_interest.shape[0])*this_stim
            x_axis_value_running_trials.append(this_stim)

        if analysis_methods.get("plotting_event_related_velocity") == True:
            fig, axes = plt.subplots(
                1, len(velocity_list), figsize=(40, 3), squeeze=False
            )
            tp = np.linspace(0, window_length, velocity_list[0].shape[1])
            for i, ax in zip(range(0, index_points.shape[1]), axes.reshape(-1)):
                this_response = velocity_list[i]
                this_coherence = x_axis_value_running_trials[i]
                mean_response = np.mean(this_response, axis=0)
                sem_response = np.std(this_response, axis=0, ddof=1) / np.sqrt(
                    this_response.shape[0]
                )
                v_add_sem = mean_response + sem_response
                v_min_sem = mean_response - sem_response
                ax.plot(tp, mean_response)
                ax.fill_between(tp, v_add_sem, v_min_sem, alpha=0.5, linewidth=0)
                ax.axvline(
                    x=abs(analysis_window[0]),
                    color="black",
                    linestyle="-",
                    label="Stim start",
                    linewidth=1,
                )
                # ax.axvline(x=(abs(analysis_window[0])+stim_duration), color='black', linestyle='--', label='Stim end',linewidth=1)
                ax.set_title(f"stimulus coherence level: {this_coherence}")
                # ax.title=(f'coherence_{this_coherence}_during_stimulus')
                # plot_file=f'coherence_{this_coherence}_during_stimulus.png'
                # plt.ylim(0.1, 100)
                # ax.set_yticks([0,10,20])
                # ax.set_yticklabels(['0','10','20'])
                # ax.set_yscale('log')

                # plt.savefig(thisDir+plot_file)
            plt.show()

        if analysis_methods.get("plotting_optomotor_response") == True:
            colormap = np.array(["r", "y", "m", "c", "k", "b", "g", "r"])
            fig, (ax, ax1) = plt.subplots(
                nrows=1, ncols=2, figsize=(18, 7), tight_layout=True
            )
            for i in range(len(opto_running_trials)):
                this_response = opto_running_trials[i]
                this_coherence = x_axis_value_running_trials[i]
                mean_response = np.mean(this_response, axis=0)
                sem_response = np.std(this_response, axis=0, ddof=1) / np.sqrt(
                    this_response.shape[0]
                )
                ax.errorbar(
                    this_coherence,
                    mean_response,
                    yerr=sem_response,
                    c=colormap[0],
                    fmt="o",
                    elinewidth=2,
                    capsize=3,
                )
            ax.set_ylim(-1, 1)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(["-1", "0", "1"], size=16)
            # ax.set_xticks([1,2,3,4,5,6,7,8,9])
            # ax.set_xticklabels(['100','75','50','25','0','25','50','75','100'],size=16)
            ax.set_xlabel("Coherence level (%)", fontsize=18)
            ax.set_ylabel("Optomotor Index", fontsize=18)
            # plt.show()
        return travel_distance_running_trials, opto_running_trials, index_list


if __name__ == "__main__":
    # thisDir='C:/Users/neuroLaptop/Documents/L23038/230824/dots'# a test example on neurolocust laptop
    thisDir = "Z:/DATA/experiment_trackball_Optomotor/L23042/230824/dots"  # a test example on the server
    json_file = "analysis_methods_dictionary.json"
    tic = time.perf_counter()
    main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
