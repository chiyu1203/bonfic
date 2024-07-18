"""spontaneous response: needs to figure out stationary, moving, timing and total travel distance as a overview 
event evoked response: needs to divide stationary trails, moving trails 
(in the case of optomotor response, need to subdivide them into moving along stimulus, 
moving against stimulus and others)
event evoked response: need to output the trial id (both within the same stimulus type and overall) and response, 
velocity vector across time in a trial plus optomotor response
analyis methods needs to tell us what to analye. For example optomotor response, acceleration epochs, deceleration epochs, transition between stationary to walk and walk to stationary
"""

import json, sys, re, time, os, fnmatch, warnings
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from pathlib import Path
from align_data_with_ttl import analyse_led_signal

##need to add this additional cell because useful tools are in another folder. Need to integrate these two folders one day
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import select_animals_gpt, find_file
from data_cleaning import sorting_trial_info


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


def a_guest(x):
    # Initialize an array of zeros with the same shape as x
    b = np.zeros_like(x, dtype=bool)

    # Find the index of the maximum value in each row of x
    i = np.argmax(x, axis=1)

    # Set the value at the index of the maximum value to True for each row of x
    b[np.arange(i.size), i] = np.logical_or.reduce(x, axis=1)

    return b


def deceleration_detection(v_array=[], *args):
    # Set initial values for variables
    init_frame = args[0]
    monitor_fps = args[1]
    presence_duration = args[2]
    fir_half = int(init_frame + 0.5 * monitor_fps)

    # Calculate mean and standard deviation of initial velocity values
    V0 = v_array[:, init_frame:fir_half]
    meanV0 = np.mean(V0, axis=1)
    stdV0 = np.std(V0, axis=1, ddof=1)

    # Calculate deceleration threshold
    deceleration_threshold = meanV0 - 2 * stdV0

    # Calculate velocity of interest and create a threshold matrix
    velocity_of_interest = v_array[
        :, fir_half : fir_half + presence_duration * monitor_fps
    ]
    dece_big = np.reshape(
        np.repeat(deceleration_threshold, presence_duration * monitor_fps, axis=0),
        (len(deceleration_threshold), presence_duration * monitor_fps),
    )

    # Check for deceleration onset and return the onset time
    dec = np.less(dece_big, velocity_of_interest)

    # In theory, chatGPT's method has no problem but it needs some additional work to sort out all True and all False condition
    # last_true_indices = dec.shape[1] - np.argmax(dec[:, ::-1], axis=1) - 1
    # Create a new array with only the first "True" value from each row
    # result_array = np.zeros(dec.shape[0], dtype=bool)
    # result_array[last_true_indices != dec.shape[1]-1] = dec[np.arange(dec.shape[0]), last_true_indices[last_true_indices != dec.shape[1]-1]]

    rev = np.fliplr(dec)
    rev_t_seq_onset = np.where(a_guest(rev) == True)
    return rev_t_seq_onset


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


def generate_stimulus_evoked_response(
    stim_to_analyse,
    vz,
    travel_distance_fbf,
    analysis_window,
    monitor_fps,
    stim_duration,
):
    print(f"plotting data from {analysis_window[0]} to {analysis_window[1]}")
    sec_thres_min = 5
    sec_thres_max = 30
    sec_steps = 5
    init_time = abs(analysis_window[0])
    init_frame = init_time * monitor_fps
    end_frame = analysis_window[1] * monitor_fps
    index_points = generate_index_points(stim_to_analyse, analysis_window, monitor_fps)
    velocity_z = vz[index_points]
    stim_travel_distance = travel_distance_fbf[index_points]
    velocity = stim_travel_distance * monitor_fps
    cum_z_vector = np.cumsum(velocity_z, axis=1)
    cum_travel_distance = np.cumsum(stim_travel_distance, axis=1)
    opto_all_trials = np.ones((velocity_z.shape[0], 5))
    opto_all_isi = np.ones((velocity_z.shape[0], 1))
    ## when quantifying optomotor response, we focus on only stimulus onset so using init_frame:end_frame
    for j, analysis_length in enumerate(range(sec_thres_min, sec_thres_max, sec_steps)):
        this_end = (init_time + analysis_length) * monitor_fps
        for i in range(0, velocity_z.shape[0]):
            opto_all_trials[i, j] = optomotor_response_idx(
                velocity_z[i, init_frame:this_end, 0]
            )
            if j == 0:
                opto_all_isi[i, j] = optomotor_response_idx(
                    velocity_z[i, :init_frame, 0]
                )
    return (
        cum_travel_distance,
        opto_all_trials,
        opto_all_isi,
        cum_z_vector,
        init_frame,
        end_frame,
        velocity,
    )


def classify_walk(arr, speed_threshold=10, consecutive_length=5):
    binary_array = arr > speed_threshold

    # Create a kernel of length 5 filled with ones
    kernel = np.ones(consecutive_length)

    # Use convolution to find the sum of 5 consecutive elements
    conv = np.convolve(binary_array, kernel, mode="valid")

    # Find the indices where the convolution result is 5 or more
    start_indices = np.where(conv >= consecutive_length)[0]

    # Adjust the indices to ensure that they point to the start of 5 or more consecutive elements
    start_indices = [
        idx
        for idx in start_indices
        if np.all(binary_array[idx : idx + consecutive_length])
    ]
    return np.asarray(start_indices)


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
    if "optoxstim" in summary_df:
        follow_trials = turning_walk_trials.loc[turning_walk_trials["optoxstim"] < 0]
        against_trials = turning_walk_trials.loc[turning_walk_trials["optoxstim"] > 0]
        spontaneous_trials = turning_walk_trials.loc[
            turning_walk_trials["optoxstim"] == 0
        ]
        tmp = follow_trials.groupby("stim_type").count()
        follow_count_coherence = tmp.index.values
        follow_count = tmp.values[:, 0]


def generate_index_points(stim_to_analyse, analysis_window, monitor_fps):
    window_length = analysis_window[1] - analysis_window[0]
    interval_onset = (
        stim_to_analyse
        + np.ones(len(stim_to_analyse), dtype=int) * analysis_window[0] * monitor_fps
    )
    analysie_window_tile = np.tile(
        np.arange(window_length * monitor_fps), (len(interval_onset), 1)
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


def main(thisDir, json_file):
    thisDir = Path(thisDir)
    ##Figuring out path and directory
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    monitor_fps = analysis_methods.get("monitor_fps")
    stim_duration = analysis_methods.get("stim_duration")
    interval_duration = analysis_methods.get("interval_duration")
    visual_paradigm_name = analysis_methods.get("experiment_name")
    fictrac_posthoc_analysis = analysis_methods.get("fictrac_posthoc_analysis")
    align_with_isi_onset = analysis_methods.get("align_with_isi_onset")
    pd_pattern = "PD*.csv"
    this_PD = find_file(thisDir, pd_pattern)
    trial_pattern = "trial*.csv"
    csv_file_directory = find_file(thisDir, trial_pattern)
    tracking_csv_pattern = "tracking_*.csv"
    tracking_csv_file_directory = find_file(thisDir, tracking_csv_pattern)
    database_pattern = "database*.pickle"
    database = find_file(thisDir, database_pattern)
    # csv_pattern = "trial_*.csv"
    # tracking_csv_pattern = "tracking_*.csv"
    # database_pattern = "*.pickle"
    colormap_name = "coolwarm"
    COL = MplColorHelper(colormap_name, 0, 8)
    sm = cm.ScalarMappable(cmap=colormap_name)

    running_speed_threshold = 50
    turning_threshold = 0.33
    df = pd.read_pickle(database)
    [exp_date, exp_hour] = csv_file_directory.stem.split("events")[1].split("T")
    exp_place = csv_file_directory.parts[3]
    # exp_place = "Zball"
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
    vz = df.loc[:, ["delta rotation vector lab z"]].values
    travel_distance_fbf = np.sqrt(
        np.add(np.square(np.diff(x_all, axis=0)), np.square(np.diff(y_all, axis=0)))
    )
    """

    velocity_fbf = travel_distance_fbf * monitor_fps

    putative_walk = classify_walk(velocity_fbf[:, 0], 5, monitor_fps * 1)
    dif_putative_walk = np.diff(putative_walk, axis=0)
    gap_len = np.where(dif_putative_walk > monitor_fps * 1)
    walk_events_end = putative_walk[gap_len]
    walk_events_start = np.concatenate(
        (
            [walk_events_end[0] - gap_len[0][0] - 1],
            walk_events_end[1:] - gap_len[0][1:] + gap_len[0][:-1],
        )
    )
    # walk_events_start = []
    # for i in range(len(walk_events_end)):
    #     if i == 0:
    #         walk_events_start.append(walk_events_end[i] - gap_len[0][i] - 1)
    #         continue
    #     else:
    #         walk_events_start.append(
    #             walk_events_end[i] - gap_len[0][i] + gap_len[0][i - 1]
    #         )

    pcm = plt.pcolormesh(np.transpose(velocity_fbf), cmap="magma", vmin=0, vmax=100)
    for i in walk_events_start:
        plt.axvline(x=i, color="w")
    plt.show()
    """
    travel_distance_whole_session = np.sum(travel_distance_fbf)
    if analysis_methods.get("plotting_trajectory") == True:
        plt.scatter(x_all, y_all, c=np.arange(len(y_all)), marker=".", alpha=0.5)
        # plt.show()

    ##here to start analying stimulus-evoked response.
    ##Firstly, load stimulus information
    # tracking_csv_file = fnmatch.filter(file, tracking_csv_pattern)
    # tracking_csv_file_directory = os.path.join(thisDir, tracking_csv_file[0])
    if tracking_csv_file_directory != None:
        tracking_info = pd.read_csv(
            tracking_csv_file_directory, delimiter="(", engine="python"
        )
        tracking_info = tracking_info.reset_index()
        tracking_info[["timestamp", "value"]] = tracking_info.iloc[:, 1].str.split(
            ", ", expand=True
        )
        tracking_info["value"] = tracking_info["value"].str.replace(
            r"[()]", "", regex=True
        )
        tracking_info["value"] = tracking_info["value"].str.replace(
            r"[,]", ".", regex=True
        )
        tracking_info["timestamp"] = tracking_info["timestamp"].str.replace(
            r"[,]", ".", regex=True
        )
        tracking_info["timestamp"] = pd.to_numeric(tracking_info["timestamp"])
        tracking_info["timestamp"] = tracking_info["timestamp"].astype("float32")
        tracking_info["value"] = pd.to_numeric(tracking_info["value"])
    # csv_file = fnmatch.filter(file, csv_pattern)
    # csv_file_directory = os.path.join(thisDir, csv_file[0])
    stim_info = pd.read_csv(csv_file_directory)
    # stim_info = stim_info.reset_index()
    raw_column_number = stim_info.shape[1]
    stim_variable = []
    stimulus_timestamp = []
    if visual_paradigm_name == "gratings":
        stim_info = stim_info.reset_index()
        for row in range(0, len(stim_info)):

            # delete the bracket in current speed and save it as integer in new list
            stim_variable.append(int(stim_info.iloc[row, 2].replace("(", "")))
            stimulus_timestamp.append(float(stim_info.iloc[row, 3].replace(")", "")))
        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))
        stim_info["stim_type"] = stim_variable
        stim_type = np.unique(stim_variable)
        # or stim_type = stim_variable if I need to use list
        # stim_type = np.sort(stim_info.stim_type.unique()).tolist()

    elif visual_paradigm_name == "sweeploom":
        stim_type = analysis_methods.get("stim_type")
        # stim_type = [
        #     "c2b_f_sloom",
        #     "c2b_s_sloom",
        #     "c2f_f_sloom",
        #     "c2f_s_sloom",
        #     "c2c_f_loom",
        #     "c2c_s_loom",
        # ]
        for row in range(0, len(stim_info)):
            stim_variable.append(int(stim_info.iloc[row, 2].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 3].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 4].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 5].split("=", 1)[1]))
            stim_variable.append(int(stim_info.iloc[row, 6].split("=", 1)[1]))
            if exp_date in ["221102", "221103"]:
                stim_variable.append(int(stim_info.iloc[row, 7].split("=", 1)[1]))
                stim_variable.append(int(stim_info.iloc[row, 8].split("=", 1)[1]))
            # stim_variable.append(int(re.split(r'=|}',stim_info.iloc[row,5])[1]))
            if isinstance(stim_info.loc[row, "Value"], float):
                timestamp = stim_info.loc[row, "Timestamp"]
            else:
                timestamp = (
                    str(stim_info.loc[row, "Timestamp"])
                    + "."
                    + stim_info.loc[row, "Value"]
                )
            stim_variable.append(float(timestamp.replace(")", "")))
        stim_variable = np.array(stim_variable).reshape((len(stim_info), -1))
        column_names = [
            "size_start",
            "size_end",
            "location1_start",
            "location1_end",
            "location2_start",
            "location2_end",
            "time",
            "timestamp",
        ]
        stim_info = pd.DataFrame(stim_variable, columns=column_names)
        if exp_date in ["2022-10-15", "2022-10-16"]:
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

    elif visual_paradigm_name == "coherence" or visual_paradigm_name.endswith(
        "densities"
    ):
        stim_info, stim_type = sorting_trial_info(
            stim_info, visual_paradigm_name="coherence", exp_place="Zball"
        )
    else:
        warnings.warn("stimulus paradigm can not be recognised")
        return

    # compare the timestamp between event file and dat file
    stimulus_timestamp = stim_info["ts"].to_numpy()
    if fictrac_posthoc_analysis:
        stim_onset, isi_onset = analyse_led_signal(thisDir, json_file)
        transition_onset = np.ones((len(stim_onset) + len(isi_onset),), dtype=int)
        transition_onset[1::2] = stim_onset
        transition_onset[::2] = isi_onset
        stim_info["stim_onset"] = transition_onset
    elif df["timestamp"][1] != df["timestamp"][2]:
        data_timestamp = df.loc[:, "timestamp"]
        stimulus_timestamp_b = np.tile(stimulus_timestamp, (len(data_timestamp), 1))
        data_timestamp_b = np.tile(data_timestamp, (len(stimulus_timestamp), 1))
        timestamp_diff = data_timestamp_b > np.transpose(stimulus_timestamp_b)
        stim_onset = np.where(np.roll(timestamp_diff, 1, axis=1) != timestamp_diff)[1][
            1::2
        ]
        stim_info["stim_onset"] = stim_onset
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
    ##probably need to add some function here to deal with the situation when isi_onset_log is True

    if visual_paradigm_name == "sweeploom":
        analysis_window = [-1, 8]
        stim_duration = 4
        presence_duration = 1
        sweep_duration = 2
        velocity_threshold = 5
    elif visual_paradigm_name == "coherence" or visual_paradigm_name.endswith(
        "densities"
    ):
        analysis_window = [interval_duration * -1, stim_duration + interval_duration]
    else:
        print(
            "visual paradigm name does not match with avaliable options. Please check"
        )
        return

    cum_travel_isi = []
    summary_list_isi = []
    velocity_isi = []
    z_isi = []
    cum_travel_list = []
    summary_df_list = []
    velocity_list = []
    z_list = []
    tick_positions = np.linspace(0, 1, len(stim_type))
    tick_labels = [str(value) for value in stim_type]
    if (
        analysis_methods.get("analysis_by_stimulus_type") == True
    ):  ## this is usually use when stimulus length is different
        for thisStim in stim_type:
            stim_to_analyse = stim_info.loc[
                np.where((stim_info.loc[:, "stim_type"] == thisStim)), "stim_onset"
            ]
            stim_name_list = stim_info.loc[stim_to_analyse.index, "stim_type"].values
            if visual_paradigm_name == "sweeploom":
                if re.search(
                    "_f_", thisStim
                ):  ##create this condition so that stim_duration became dynamics
                    stim_duration = 3
                    presence_duration = 1
                    sweep_duration = 1
                elif re.search("_s_", thisStim):
                    stim_duration = 4
                    presence_duration = 1
                    sweep_duration = 2
                else:
                    print("something wrong with the stim list")
            (
                cum_travel_distance,
                opto_all_trials,
                opto_all_isi,
                cum_z_vector,
                init_frame,
                end_frame,
                velocity,
            ) = generate_stimulus_evoked_response(
                stim_to_analyse,
                vz,
                travel_distance_fbf,
                analysis_window,
                monitor_fps,
                stim_duration,
            )
            travel_distance_prior_stim = cum_travel_distance[:, init_frame, 0]
            travel_distance_during_stim = (
                cum_travel_distance[:, end_frame, 0] - travel_distance_prior_stim
            )
            summary_list = {
                "opto_index_isi": opto_all_isi[:, 0],
                "opto_index": opto_all_trials,
                "travel_distance_isi": travel_distance_prior_stim,
                "travel_distance": travel_distance_during_stim,
                "stim_index": stim_to_analyse.index,
                "stim_type": stim_name_list,
            }
            summary_df = pd.DataFrame(summary_list)
            classify_trial_type(summary_df, turning_threshold, running_speed_threshold)
            cum_travel_list.append(cum_travel_distance[:, :, 0])
            summary_df_list.append(summary_df)
            velocity_list.append(velocity[:, :, 0])
            z_list.append(cum_z_vector[:, :, 0])
    else:
        if align_with_isi_onset:
            stim_to_analyse = stim_info.loc[::2, "stim_onset"]
            analysis_window_isi = [
                interval_duration * -1,
                interval_duration + interval_duration,
            ]
            stim_name_list = np.array(stim_info["stim_type"][stim_to_analyse.index])
            (
                cum_travel_distance,
                opto_all_trials,
                opto_all_isi,
                cum_z_vector,
                init_frame_isi,
                end_frame_isi,
                velocity,
            ) = generate_stimulus_evoked_response(
                stim_to_analyse,
                vz,
                travel_distance_fbf,
                analysis_window_isi,
                monitor_fps,
                interval_duration,
            )
            travel_distance_prior_stim = cum_travel_distance[:, init_frame_isi, 0]
            travel_distance_during_stim = (
                cum_travel_distance[:, end_frame_isi, 0] - travel_distance_prior_stim
            )
            optomotor_range = 1
            summary_isi = {
                "opto_index_last5": opto_all_isi[:, 0],
                "opto_index_isi": opto_all_trials[:, optomotor_range],
                "travel_distance_last5": travel_distance_prior_stim,
                "travel_distance_isi": travel_distance_during_stim,
                "stim_index": stim_to_analyse.index,
                "stim_type": stim_name_list,
            }
            summary_df_isi = pd.DataFrame(summary_isi)
            # classify_trial_type(summary_df, turning_threshold, running_speed_threshold)
            cum_travel_isi.append(cum_travel_distance[:, :, 0])
            summary_list_isi.append(summary_df_isi)
            velocity_isi.append(velocity[:, :, 0])
            z_isi.append(cum_z_vector[:, :, 0])
        else:
            stim_to_analyse = stim_info.loc[1::2, "stim_onset"]
            stim_name_list = np.array(stim_info["stim_type"][stim_to_analyse.index])
            (
                cum_travel_distance,
                opto_all_trials,
                opto_all_isi,
                cum_z_vector,
                init_frame,
                end_frame,
                velocity,
            ) = generate_stimulus_evoked_response(
                stim_to_analyse,
                vz,
                travel_distance_fbf,
                analysis_window,
                monitor_fps,
                stim_duration,
            )
            travel_distance_prior_stim = cum_travel_distance[:, init_frame, 0]
            travel_distance_during_stim = (
                cum_travel_distance[:, end_frame, 0] - travel_distance_prior_stim
            )
            optomotor_range = 3
            summary_list = {
                "opto_index_isi": opto_all_isi[:, 0],
                "opto_index": opto_all_trials[:, optomotor_range],
                "travel_distance_isi": travel_distance_prior_stim,
                "travel_distance": travel_distance_during_stim,
                "stim_index": stim_to_analyse.index,
                "stim_type": stim_name_list,
            }
            summary_df = pd.DataFrame(summary_list)
            classify_trial_type(summary_df, turning_threshold, running_speed_threshold)
            cum_travel_list.append(cum_travel_distance[:, :, 0])
            summary_df_list.append(summary_df)
            velocity_list.append(velocity[:, :, 0])
            z_list.append(cum_z_vector[:, :, 0])

    ##concatenate travel distance for histogram

    print("ready to plot graph with some data and decompose those lists")
    ##concatenate velocity list for heatmap
    if align_with_isi_onset:
        summary_df = pd.concat(cum_travel_isi)
        cum_travel_distance = np.vstack(summary_list_isi)
        velocity_arr = np.vstack(velocity_isi)
        cum_z_vector = np.vstack(z_isi)
    else:
        summary_df = pd.concat(summary_df_list)
        cum_travel_distance = np.vstack(cum_travel_list)
        velocity_arr = np.vstack(velocity_list)
        cum_z_vector = np.vstack(z_list)

    if analysis_methods.get("plotting_tbt_overview") == True:
        ##arrange xticks at onset and offset of the stimulus for heatmap
        # trial_num_across_stim=np.cumsum(trial_num)
        fig1, (ax, ax1, ax2) = plt.subplots(
            nrows=1, ncols=3, figsize=(18, 7), tight_layout=True
        )
        # ax.hist(optomotor_response_during_stim, bins=20)
        ax.hist(summary_df["opto_index"].values, bins=20)
        ax.set_xlabel("optomotor index")
        ax.set_ylabel("frequency")
        ax.set_title("optomotor response during stimulus presentation")
        # ax1.hist(travel_distance_during_stim, bins=20)
        ax1.hist(summary_df["travel_distance"].values, bins=20)
        ax1.set_xlabel("travel distance (mm)")
        ax1.set_ylabel("frequency")
        ax1.set_title("travel distance during stimulus presentation")
        ## draw subplot2
        pcm = ax2.pcolormesh(velocity_arr, cmap="magma", vmin=0, vmax=100)
        # pcm=ax2.pcolormesh(velocity_tbt, cmap = 'magma') ##sometimes, we dont need a vmax
        ## add colorbar
        fig1.colorbar(pcm)
        ax2.set_yticks([])
        ax2.set_ylabel("Trial#")
        ## add vertical lines to indicate stimulus onset and offset
        ax2.axvline(
            x=init_frame,
            color="white",
            linestyle="--",
            label="Stimulus Starts",
        )
        ax2.axvline(
            x=end_frame,
            color="white",
            linestyle="--",
            label="Stimulus Ends",
        )
        ax2.set_xticks([init_frame])
        ax2.set_xticklabels(["0"])
        ax2.set_xlabel("Time (s)")
        ## add title
        ax2.set_title("stimulus evoked velocity preview")
        plot_name = "optomotor_overview.png"
        if analysis_methods.get("debug_mode") == False:
            fig1.savefig(Path(thisDir) / plot_name)
        plt.show()

    if analysis_methods.get("plotting_optomotor_response") == True:
        # colormap = np.array(["k", "y", "m", "c", "r", "b", "g", "k"])
        fig2, (ax3, ax4) = plt.subplots(
            nrows=1, ncols=2, figsize=(18, 7), tight_layout=True
        )
        for i in range(len(stim_type)):
            this_coherence = stim_type[i]
            this_response = summary_df.loc[
                summary_df["stim_type"] == this_coherence, "opto_index"
            ].values
            if len(this_response) == 0:
                continue
            mean_response = np.mean(this_response, axis=0)
            sem_response = np.std(this_response, axis=0, ddof=1) / np.sqrt(
                this_response.shape[0]
            )
            ax3.errorbar(
                this_coherence,
                mean_response,
                yerr=sem_response,
                c="k",
                fmt="o",
                elinewidth=2,
                capsize=3,
            )
            this_z = cum_z_vector[summary_df["stim_type"] == this_coherence, :]
            time_points = np.arange(this_z.shape[1])
            time_points_b = np.tile(time_points, (this_z.shape[0], 1))
            # ax4.scatter(time_points_b, this_z, s=1, color=custom_cmap[i])
            ax4.scatter(time_points_b, this_z, s=1, color=COL.get_rgb(i))

        ax3.set_ylim(-1, 1)
        ax3.set(
            xlabel=visual_paradigm_name,
            yticks=[-1, 0, 1],
            ylabel="Optomotor Index during stimulus",
        )
        # xticks=[np.max(stim_type), 0, np.min(stim_type)],
        ax4.axvline(
            x=init_frame,
            color="black",
            linestyle="--",
            label="Stimulus Starts",
        )
        ax4.axvline(
            x=end_frame,
            color="black",
            linestyle="--",
            label="Stimulus Ends",
        )
        ax4.set_ylim(-14400, 14400)
        ax4.set(
            yticks=[-14400, 0, 14400],
            ylabel="Cumulative Rotation Velocity(sec)",
            xticks=[0, init_frame, end_frame],
            xticklabels=[str(analysis_window[0]), "0", str(stim_duration)],
            xlabel="Time (s)",
        )
        cbar = plt.colorbar(sm, ax=ax4, orientation="vertical", ticks=tick_positions)
        cbar.set_ticklabels(tick_labels)

        plot_name = f"optomotor_analysis_until{(optomotor_range+1)*5}s.png"
        if analysis_methods.get("debug_mode") == False:
            fig2.savefig(Path(thisDir) / plot_name)
        plt.show()

    if analysis_methods.get("plotting_deceleration_accerleration") == True:
        travel_distance_after_stim_presence = cum_travel_distance[
            :, init_frame + presence_duration * monitor_fps
        ]
        travel_distance_1sec_prior_stim = (
            cum_travel_distance[:, init_frame] - cum_travel_distance[:, 0]
        )
        running_trial = travel_distance_1sec_prior_stim > velocity_threshold
        running_velocity = velocity_arr[running_trial, :]
        running_travel_distance_after_stim_presence = (
            travel_distance_after_stim_presence[running_trial]
        )
        travel_distance_1sec_post_stim = (
            cum_travel_distance[running_trial, init_frame + 1 * monitor_fps]
            - cum_travel_distance[running_trial, init_frame]
        )
        run_type1 = running_velocity[
            travel_distance_1sec_post_stim > velocity_threshold, :
        ]
        running_travel_distance_after_stim_presence_type1 = (
            running_travel_distance_after_stim_presence[
                travel_distance_1sec_post_stim > velocity_threshold
            ]
        )
        trial_to_analyse = run_type1
        detected_bool = deceleration_detection(
            trial_to_analyse, init_frame, monitor_fps, presence_duration
        )
        dec_trial_id = detected_bool[0]
        # this reverse the frame number back so the output is the frame number no. at the detection window
        de_test = monitor_fps * presence_duration - detected_bool[1]
        ##remove trials that the true value is outside this 1-sec window
        dev_timing = de_test[de_test != monitor_fps * presence_duration]
        dev_trial_id = dec_trial_id[de_test != monitor_fps * presence_duration]
        dev_timing_from_analysis_window = (
            dev_timing
            + init_frame
            + (0.5 * presence_duration + abs(analysis_window[0])) * monitor_fps
        )
        # dev_timing_from_stim_presence = (
        #     dev_timing / monitor_fps + 0.5 * presence_duration + init_time
        # )
        # dev_timing_y = 10 * np.ones_like(dev_timing_from_stim_presence)
        ##When analysing recover epochs, focus on the trials has similar deceleration dynamics
        dev_trial = trial_to_analyse[dev_trial_id, :]
        ##apply these velocity traces with velocity threshold
        index_list = recovery_detection(dev_trial, dev_timing_from_analysis_window)
        rec_array = (
            index_list - dev_timing_from_analysis_window
        )  # this normalise the recovery duration from the first deceleration

    if analysis_methods.get("debug_mode") == True:
        print("Nothing is saved in the debug mode. Move to the next animal")
    else:
        summary_file_name = Path(thisDir) / "behavioural_summary.pickle"
        summary_df.to_pickle(summary_file_name)
        velocity_file_name = Path(thisDir) / "trial_velocity.npy"
        with open(velocity_file_name, "wb") as f:
            np.save(f, velocity_arr)
        Zvector_file_name = Path(thisDir) / "z_vector.npy"
        with open(Zvector_file_name, "wb") as f:
            np.save(f, cum_z_vector)

    return summary_df, velocity_arr, cum_z_vector, travel_distance_whole_session


"""
need to write something here to save summary_df,velocity_arr, cum_z_vector, but what is the efficent way to manage these data
If everything is in summary_df then that's easy but now it looks like I have other two big tables velocity_arr and cum_z_vector.
or should I export the original location and then do the calculation after aligning behaviour data with ephys data
My gut feeling is that we still have a place for hdf-based curated database, which we can throw all the information there
In this case, the original database_curated.pickle should be renamed as tracking_curated.pickle.
"""


if __name__ == "__main__":
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/L22131/221102/sweeploom"
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/VCCball/L22036/220819/densities"
    # thisDir = ("Z:/DATA/experiment_trackball_Optomotor/VCCball/L23002/230221/long_densities")
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/VCCball/L23004/230218/densities"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23001\231121\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23012\231126\coherence\session1"
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\locustMatrex\GN24001\240713\gratings\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23001\231121\coherence\session1"
    # thisDir = (
    #     "Z:/DATA/experiment_trackball_Optomotor/Zball/GN23019/240507/coherence/session1"
    # )
    # thisDir = r"Z:/DATA/experiment_trackball_Optomotor/VCCball/L23009/230309/densities"
    # thisDir = "/Users/cylee/Downloads/bonfic_led"
    # json_file = "analysis_methods_dictionary.json"
    json_file = r"C:\Users\neuroPC\Documents\GitHub\UnityDataAnalysis\analysis_methods_dictionary.json"
    tic = time.perf_counter()
    main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
