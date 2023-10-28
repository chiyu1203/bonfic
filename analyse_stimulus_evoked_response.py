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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm


# custom_cmap = [
#     [1, 0, 0, 1],
#     [0.8, 0, 0.2, 0.1],
#     [0.75, 0, 0.25, 0.05],
#     [0.7, 0, 0.3, 0.02],
#     [0.5, 0, 0.5, 0.02],
#     [0.3, 0, 0.7, 0.02],
#     [0.25, 0, 0.75, 0.05],
#     [0.2, 0, 0.8, 0.1],
#     [0, 0, 1, 1],
# ]
class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def optomotor_response_idx(ev_dat):
    opt_idx = (
        np.nansum(abs(ev_dat[ev_dat > 0])) - np.nansum(abs(ev_dat[ev_dat < 0]))
    ) / (np.nansum(abs(ev_dat[ev_dat < 0])) + np.nansum(abs(ev_dat[ev_dat > 0])))
    return opt_idx


def main(thisDir, json_file):
    csv_pattern = "trial_*.csv"
    tracking_csv_pattern = "tracking_*.csv"
    database_pattern = "*.pickle"
    frame_rate = 144
    stim_duration = 25
    grey_duration = 5
    # analysis_window=frame_rate*(stim_duration+grey_duration)
    analysis_window = [grey_duration * -1, stim_duration]
    running_speed_threshold = 50
    turning_threshold = 0.33
    window_length = analysis_window[1] - analysis_window[0]
    stim_type = [
        -100,
        -75,
        -50,
        -25,
        0,
        25,
        50,
        75,
        100,
    ]  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
    colormap_name = "coolwarm"
    COL = MplColorHelper(colormap_name, 0, 8)
    sm = cm.ScalarMappable(cmap=colormap_name)
    tick_positions = np.linspace(0, 1, 9)
    tick_labels = [str(value) for value in stim_type]
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
        # instant_v = savgol_filter(df.loc[:, ["instant speed"]], 39, 3, axis=0)
        # runmean_v = df["instant speed"].rolling(10).mean().values
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
    tracking_csv_file = fnmatch.filter(file, tracking_csv_pattern)
    tracking_csv_file_directory = os.path.join(thisDir, tracking_csv_file[0])
    tracking_info = pd.read_csv(
        tracking_csv_file_directory, delimiter="(", engine="python"
    )
    tracking_info = tracking_info.reset_index()
    tracking_info[["timestamp", "value"]] = tracking_info.iloc[:, 1].str.split(
        ", ", expand=True
    )
    tracking_info["value"] = tracking_info["value"].str.replace(r"[()]", "", regex=True)
    tracking_info["value"] = tracking_info["value"].str.replace(r"[,]", ".", regex=True)
    tracking_info["timestamp"] = tracking_info["timestamp"].str.replace(
        r"[,]", ".", regex=True
    )
    tracking_info["timestamp"] = pd.to_numeric(tracking_info["timestamp"])
    tracking_info["timestamp"] = tracking_info["timestamp"].astype("float32")
    tracking_info["value"] = pd.to_numeric(tracking_info["value"])
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
        stim_info["stim_type"] = np.select(filters, stim_type)

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
        stim_info = pd.DataFrame(stim_variable)
        if exp_date in ["221015", "221016"]:
            ### use this criteria for experiments during exp_date=['221015','221016']
            filters = [
                (stim_info.iloc[:, 0] == 1)
                & (stim_info.iloc[:, 2] > stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 8)
                & (stim_info.iloc[:, 2] > stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 1)
                & (stim_info.iloc[:, 2] < stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 8)
                & (stim_info.iloc[:, 2] < stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 1)
                & (stim_info.iloc[:, 2] == stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 0] == 8)
                & (stim_info.iloc[:, 2] == stim_info.iloc[:, 3]),
            ]

        else:
            filters = [
                (stim_info.iloc[:, 6] == 1)
                & (stim_info.iloc[:, 2] > stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 2)
                & (stim_info.iloc[:, 2] > stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 1)
                & (stim_info.iloc[:, 2] < stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 2)
                & (stim_info.iloc[:, 2] < stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 1)
                & (stim_info.iloc[:, 2] == stim_info.iloc[:, 3]),
                (stim_info.iloc[:, 6] == 2)
                & (stim_info.iloc[:, 2] == stim_info.iloc[:, 3]),
            ]
        stim_info["stim_type"] = np.select(filters, stim_type)
    elif analysis_methods.get("experiment_name") == "dots":
        col_num = [2, 4, 5, 7, 11, 13, 15, 17]
        col_name = [
            "ID",
            "Size",
            "Colour",
            "numDots",
            "Contrast",
            "VelX",
            "VelY",
            "Coherence",
        ]
        for i, j in zip(col_num, col_name):
            stim_info[["tmp", j]] = stim_info.iloc[:, i].str.split("=", expand=True)
            stim_info[j] = pd.to_numeric(stim_info[j])
        stim_info["Value"] = stim_info["Value"].str.replace(r"[()]", "", regex=True)
        if stim_info["Timestamp"].dtypes == "object":
            stim_info["Timestamp"] = stim_info["Timestamp"].str.replace(
                r"[()]", "", regex=True
            )
        stim_info["Value"] = stim_info["Value"].fillna(0)
        stim_info["Timestamp"] = stim_info["Timestamp"].astype(str)
        stim_info["Value"] = stim_info["Value"].astype(str)
        stim_info["timestamp"] = stim_info[["Timestamp", "Value"]].agg(".".join, axis=1)
        stim_info["timestamp"] = pd.to_numeric(stim_info["timestamp"])
        stim_info["timestamp"] = stim_info["timestamp"].astype("float32")
        stim_info.drop(columns=stim_info.columns[0:22], inplace=True)
        ##define the category of stimulus, this is used when the category is not numeric such as loom sweep-loom
        # filters = [
        #     (stim_info["VelX"] > 1) & (stim_info["Coherence"] == 100),
        #     (stim_info["VelX"] > 1) & (stim_info["Coherence"] == 75),
        #     (stim_info["VelX"] > 1) & (stim_info["Coherence"] == 50),
        #     (stim_info["VelX"] > 1) & (stim_info["Coherence"] == 25),
        #     (stim_info["VelX"] < 1) & (stim_info["Coherence"] == 0),
        #     (stim_info["VelX"] < 1) & (stim_info["Coherence"] == 25),
        #     (stim_info["VelX"] < 1) & (stim_info["Coherence"] == 50),
        #     (stim_info["VelX"] < 1) & (stim_info["Coherence"] == 75),
        #     (stim_info["VelX"] < 1) & (stim_info["Coherence"] == 100),
        # ]
        velxcoherence = (
            stim_info["VelX"] * stim_info["Coherence"] / abs(stim_info["VelX"])
        )
        stim_info["stim_type"] = -1 * velxcoherence.astype(int)

    else:
        warnings.warn("stimulus paradigm can not be recognised")
        return

    # compare the timestamp between event file and dat file
    stimulus_timestamp = stim_info["timestamp"].to_numpy()
    if df.loc[:, "timestamp"][0] > 0:
        data_timestamp = df.loc[:, "timestamp"]
    else:
        data_timestamp = tracking_info["timestamp"]
        stimulus_timestamp_b = np.tile(stimulus_timestamp, (len(data_timestamp), 1))
        data_timestamp_b = np.tile(data_timestamp, (len(stimulus_timestamp), 1))
        timestamp_diff = data_timestamp_b > np.transpose(stimulus_timestamp_b)
        stim_onset1 = np.where(np.roll(timestamp_diff, 1, axis=1) != timestamp_diff)[1][
            1::2
        ]
        warnings.warn(
            "still need to debug this place for situation where there is no log file"
        )
        return

    stimulus_timestamp_b = np.tile(stimulus_timestamp, (len(data_timestamp), 1))
    data_timestamp_b = np.tile(data_timestamp, (len(stimulus_timestamp), 1))
    timestamp_diff = data_timestamp_b > np.transpose(stimulus_timestamp_b)
    stim_onset = np.where(np.roll(timestamp_diff, 1, axis=1) != timestamp_diff)[1][1::2]
    stim_info["stim_onset"] = stim_onset
    travel_distance_during_stim_all = []
    travel_distance_prior_stim_all = []
    optomotor_response_during_stim_all = []
    velocity_list = []
    init_time = abs(analysis_window[0])
    init_frame = init_time * frame_rate
    end_time = abs(analysis_window[0]) + stim_duration
    end_frame = end_time * frame_rate
    vz = df.loc[:, ["delta rotation vector lab z"]].values
    stim_to_analyse = stim_info.loc[:, "stim_onset"]
    stim_name_list = np.array(stim_info["stim_type"])
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
    cum_z_vector = np.cumsum(velocity_z[:, init_frame:, 0], axis=1)

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
    turning_trials = summary_df.loc[abs(summary_df["opto_index"]) > turning_threshold]
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
    follow_trials = turning_walk_trials.loc[turning_walk_trials["optoxcoherence"] < 0]
    against_trials = turning_walk_trials.loc[turning_walk_trials["optoxcoherence"] > 0]
    spontaneous_trials = turning_walk_trials.loc[
        turning_walk_trials["optoxcoherence"] == 0
    ]
    tmp = follow_trials.groupby("coherence_level").count()
    follow_count_coherence = tmp.index.values
    follow_count = tmp.values[:, 0]

    travel_distance_during_stim_all.append(travel_distance_during_stim)
    travel_distance_prior_stim_all.append(travel_distance_prior_stim)
    optomotor_response_during_stim_all.append(opto_all_trials)
    # travel_distance_tbt=np.concatenate(travel_distance_prior_stim_all)
    ##concatenate travel distance for histogram
    print("ready to plot graph with some data")

    if analysis_methods.get("plotting_tbt_overview") == True:
        optomotor_response_during_stim = np.concatenate(
            optomotor_response_during_stim_all
        )
        travel_distance_prior_stim = np.concatenate(travel_distance_prior_stim_all)
        travel_distance_during_stim = np.concatenate(travel_distance_during_stim_all)

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
        ax2.set_yticks([])
        ax2.set_ylabel("Trial#")
        ## add vertical lines to indicate stimulus onset and offset
        ax2.axvline(
            x=heatmap_xticks[0],
            color="white",
            linestyle="-",
            label="Stimulus Starts",
        )
        ax2.set_xticks([heatmap_xticks[0]])
        ax2.set_xticklabels(["0"])
        ax2.set_xlabel("Time (s)")
        ## add title
        ax2.set_title("stimulus evoked velocity preview")
        plot_name = "optomotor_overview.png"
        fig1.savefig(os.path.join(thisDir, plot_name))

    if analysis_methods.get("plotting_optomotor_response") == True:
        colormap = np.array(["k", "y", "m", "c", "r", "b", "g", "k"])
        fig2, (ax3, ax4) = plt.subplots(
            nrows=1, ncols=2, figsize=(18, 7), tight_layout=True
        )
        for i in range(len(stim_type)):
            this_coherence = stim_type[i]
            this_response = summary_df.loc[
                summary_df["coherence_level"] == this_coherence, "opto_index"
            ].values
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
            this_z = cum_z_vector[summary_df["coherence_level"] == this_coherence, :]
            time_points = np.arange(this_z.shape[1])
            time_points_b = np.tile(time_points, (this_z.shape[0], 1))
            # ax4.scatter(time_points_b, this_z, s=1, color=custom_cmap[i])
            ax4.scatter(time_points_b, this_z, s=1, color=COL.get_rgb(i))

        ax3.set_ylim(-1, 1)
        # ax3.set_yticks([-1, 0, 1])
        ax3.set(
            yticks=[-1, 0, 1],
            ylabel="Optomotor Index",
            xticks=[100, 50, 0, -50, -100],
            xlabel="Coherence level (%)",
        )
        ax4.set_ylim(-100, 100)
        ax4.set(
            yticks=[-100, 0, 100],
            ylabel="Cumulative Rotation Vector (rad/frame)",
            xticks=[0, this_z.shape[1]],
            xticklabels=["0", "25"],
            xlabel="Time (s)",
        )
        cbar = plt.colorbar(sm, ax=ax4, orientation="vertical", ticks=tick_positions)
        cbar.set_ticklabels(tick_labels)

        plot_name = "optomotor_analysis.svg"
        fig2.savefig(os.path.join(thisDir, plot_name))

    plt.show()
    return summary_df, velocity_list, travel_distance_during_stim_all, cum_z_vector


if __name__ == "__main__":
    # thisDir='C:/Users/neuroLaptop/Documents/L23038/230824/dots'# a test example on neurolocust laptop
    thisDir = "Z:/DATA/experiment_trackball_Optomotor/L22147/221106/dots"  # a test example on the server
    json_file = "analysis_methods_dictionary.json"
    tic = time.perf_counter()
    main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
