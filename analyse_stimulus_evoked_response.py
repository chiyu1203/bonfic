"""spontaneous response: needs to figure out stationary, moving, timing and total travel distance as a overview 
event evoked response: needs to divide stationary trails, moving trails 
(in the case of optomotor response, need to subdivide them into moving along stimulus, 
moving against stimulus and others)
event evoked response: need to output the trial id (both within the same stimulus type and overall) and response, 
velocity vector across time in a trial plus optomotor response
analyis methods needs to tell us what to analye. For example optomotor response, acceleration epochs, deceleration epochs, transition between stationary to walk and walk to stationary
"""

import json, sys, re, time, warnings, gzip
import pandas as pd
import numpy as np
from scipy.io import savemat
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.transforms as mt
from pathlib import Path
from align_data_with_ttl import analyse_led_signal

##need to add this additional cell because useful tools are in another folder. Need to integrate these two folders one day
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file, find_nearest, calculate_fft
from data_cleaning import sorting_trial_info, load_temperature_data,euclidean_distance


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
    end_frame = analysis_window[1] * camera_fps
    duration_for_optomotor_index=analysis_methods.get("duration_for_optomotor_index")
    yaw_rotation_tbt = extract_stim_locked_data(
        stim_to_analyse, yaw_angular_velocity[:, 0], analysis_window, camera_fps
    )
    travel_distance_tbt= extract_stim_locked_data(
        stim_to_analyse,
        travel_distance_fbf[:, 0],
        analysis_window,
        camera_fps,
    )
    if analysis_methods.get("temperature_data_options").lower() == "trial":
        if analysis_methods.get("experiment_name").lower() !="coherence":
            temp_tbt=stimulus_info['temperature'].values
            humid_tbt=stimulus_info['humidity'].values
        else:
            temp_tbt=stimulus_info['temperature'][1::2].values
            humid_tbt=stimulus_info['humidity'][1::2].values
    elif len(tem_df_this_exp) > 0:
        temp_tbt= extract_stim_locked_data(
            stim_to_analyse,
            tem_df_this_exp[:, 0],
            analysis_window,
            camera_fps,
        )
        humid_tbt= extract_stim_locked_data(
            stim_to_analyse,
            tem_df_this_exp[:, 1],
            analysis_window,
            camera_fps,
        )
        temp_tbt=np.nanmean(temp_tbt, axis=1),
        humid_tbt=np.nanmean(humid_tbt, axis=1),
    else:
        temp_tbt = np.empty((stimulus_info.shape[0], 1))
        temp_tbt[:] = np.nan
        humid_tbt = np.empty((stimulus_info.shape[0], 1))
        humid_tbt[:] = np.nan

    #######Note: opto_index_isi and travel_distance is defined based on responses during the 5 sec window prior the stimulus;
    ####### whereas the summarised jaw, velocity arrays include values starting from minimal length of ISI prior to the stimulus
    velocity = travel_distance_tbt * camera_fps  # over 1.0 is probably a jump
    cum_travel_distance = np.nancumsum(travel_distance_tbt, axis=1)
    shortened_isi=5 ##set an arbitrary time window prior to the onset of stimulus for the response to ISI. This arbitary window is needed because the duration of ISI is not stored in the dataset.
    #And then the minimal length of ISI used is usually 5 sec. Therefore, setting this duration to 5 should not cover the previous trial.
    ## Note: if the first element in the time window is shorter than this 5 sec, then just use how long it is set in the time window
    if shortened_isi*camera_fps<init_frame:
        travel_distance_prior_stim = (
            cum_travel_distance[:, init_frame]
            - cum_travel_distance[:, shortened_isi * camera_fps]
        )  ##remove the initial five sec during isi
    else:
        travel_distance_prior_stim = cum_travel_distance[:, init_frame]##this calculate the travel distance fro frame 0 to the start of the stimulus onset
        shortened_isi=abs(analysis_window[0])
    travel_distance_during_stim = (
        cum_travel_distance[:, end_frame] - cum_travel_distance[:, init_frame]
    )
    if duration_for_optomotor_index>=0:
        opto_all_trials, opto_all_isi = probes_optomotor_index(
        yaw_rotation_tbt, init_frame,end_frame,camera_fps, shortened_isi
    )
        summary_list = {
            "opto_index_isi": opto_all_isi,
            "opto_index": opto_all_trials,
            "travel_distance_isi": travel_distance_prior_stim,
            "travel_distance": travel_distance_during_stim,
            "stim_index": stim_to_analyse.index,
            "temperature": temp_tbt,
            "humidity": humid_tbt,
        }
    else:
        summary_list={
            "travel_distance_isi": travel_distance_prior_stim,
            "travel_distance": travel_distance_during_stim,
            "stim_index": stim_to_analyse.index,
            "temperature": temp_tbt,
            "humidity":humid_tbt
        }

    summary_df = pd.DataFrame(summary_list)  
    return (
        summary_df,
        cum_travel_distance,
        velocity,
        yaw_rotation_tbt,
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


def deceleration_detection(v_array=[], *args):
    # Set initial values for variables
    init_frame = args[0]
    camera_fps = args[1]
    presence_duration = args[2]
    fir_half = int(init_frame + 0.5 * camera_fps)

    # Calculate mean and standard deviation of initial velocity values
    V0 = v_array[:, init_frame:fir_half]
    meanV0 = np.mean(V0, axis=1)
    stdV0 = np.std(V0, axis=1, ddof=1)

    # Calculate deceleration threshold
    deceleration_threshold = meanV0 - 2 * stdV0

    # Calculate velocity of interest and create a threshold matrix
    velocity_of_interest = v_array[
        :, fir_half : fir_half + presence_duration * camera_fps
    ]
    dece_big = np.reshape(
        np.repeat(deceleration_threshold, presence_duration * camera_fps, axis=0),
        (len(deceleration_threshold), presence_duration * camera_fps),
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


def probes_optomotor_index(yaw_rotation_tbt, init_frame,end_frame, camera_fps, shortened_isi):
    isi_frame = init_frame - camera_fps * shortened_isi
    opto_all_trials = np.ones((yaw_rotation_tbt.shape[0]))
    opto_all_isi = np.ones((yaw_rotation_tbt.shape[0]))
    ## when quantifying optomotor response, we focus on only stimulus onset so using init_frame:end_frame
    for i in range(0, yaw_rotation_tbt.shape[0]):
        opto_all_trials[i] = optomotor_response_idx(
            yaw_rotation_tbt[i, init_frame:end_frame]
        )
        opto_all_isi[i] = optomotor_response_idx(
            yaw_rotation_tbt[i, isi_frame:init_frame]
        )
    return opto_all_trials, opto_all_isi


##### this works and should be faster than extract_events but for some reason this always miss the last consecutive_length-1 elements
# for example if the correct one is 169:181, this function will return 169:177
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
    monitor_fps = analysis_methods.get("monitor_fps")
    camera_fps = analysis_methods.get("camera_fps")
    stim_duration = analysis_methods.get("stim_duration")
    interval_duration= analysis_methods.get("interval_duration")
    visual_paradigm_name = analysis_methods.get("experiment_name")
    fictrac_posthoc_analysis = analysis_methods.get("fictrac_posthoc_analysis")
    align_with_isi_onset = analysis_methods.get("align_with_isi_onset")
    exp_place = analysis_methods.get("exp_place")
    temperature_data_options = analysis_methods.get("temperature_data_options")
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset")
    mark_jump_as_nan = analysis_methods.get("mark_jump_as_nan", False)
    save_output = analysis_methods.get("save_output")
    duration_for_optomotor_index=analysis_methods.get("duration_for_optomotor_index",0)
    yaw_axis=analysis_methods.get("yaw_axis","z").lower()# define which axis to use to examine animal's rotational velocity
    ###Think about whether to specify which rotational axis to calculate optomotor index
    ###Maybe ensure velocity can go negative, which indicates animals move backward
    # Here to set some environmental variables
    colormap_name = "coolwarm"
    sm = cm.ScalarMappable(cmap=colormap_name)
    running_speed_threshold = 50
    turning_threshold = 0.33
    # Here to allocate files
    pd_start="meta"
    #pd_ext = "lux*.csv"
    pd_ext = f"{pd_start}*.csv"
    this_PD_directory = find_file(thisDir, pd_ext)
    trial_pattern = "trial*.csv"
    csv_file_directory = find_file(thisDir, trial_pattern)
    tracking_csv_pattern = "tracking_*.csv"
    tracking_csv_file_directory = find_file(thisDir, tracking_csv_pattern)
    database_pattern = "database*.pickle"
    database = find_file(thisDir, database_pattern)
    database_pattern ="database*.parquet.gzip"
    database2 = find_file(thisDir, database_pattern)
    temperature_pattern = "temperature_humidity.npy.gz"
    this_temp_directory = find_file(thisDir, temperature_pattern)
    if int(camera_fps/2)%2==0:
        filter_window_length=int(camera_fps/2)+1
    else:
        filter_window_length=int(camera_fps/2)
    #df = pd.read_pickle(database)
    df=pd.read_parquet(database2)
    if analysis_methods.get("filtering_method") == "sg_filter":
        x_all = savgol_filter(df.loc[:, ["intergrated x position"]], filter_window_length, 3, axis=0)
        y_all = savgol_filter(df.loc[:, ["intergrated y position"]], filter_window_length, 3, axis=0)
        yaw_angular_velocity = (
        df.loc[:, [f"delta rotation vector lab {yaw_axis}"]].values
    )  
    elif analysis_methods.get("filtering_method") == "rolling_median":
        x_all =df.loc[:, ["intergrated x position"]].rolling(filter_window_length).median()
        y_all = df.loc[:, ["intergrated y position"]].rolling(filter_window_length).median()
        yaw_angular_velocity = (
        df.loc[:, [f"delta rotation vector lab {yaw_axis}"]].rolling(filter_window_length).median().values
    )  
    else:
        x_all = df.loc[:, ["intergrated x position"]]
        y_all = df.loc[:, ["intergrated y position"]]
        yaw_angular_velocity = (
        df.loc[:, [f"delta rotation vector lab {yaw_axis}"]].values
    )  
    travel_distance_fbf = np.transpose(euclidean_distance(np.transpose(x_all),np.transpose(y_all)))
    ## here write an additional function to classify stationary, moving and total travel distance
    if mark_jump_as_nan:
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
    travel_distance_whole_session = np.nansum(travel_distance_fbf)
    if analysis_methods.get("plotting_trajectory") == True:
        fig = plt.figure()  # a new figure window
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.scatter(x_all, y_all, c=np.arange(len(y_all)), marker=".", alpha=0.5)
        fig.savefig(thisDir / "trajectory_plot.png")

    # Here to load time based on one of the files created by bonsai such as PD file or events files
    if this_PD_directory:
        file_name = this_PD_directory.stem
        if pd_start=='meta':
            [_, exp_time]=re.split("meta_", file_name)
        else:
            [_, exp_time] = re.split(f"{pd_start}\d_", file_name)

        [exp_date, exp_hour] = re.split("T", exp_time)
        test_dt = pd.to_datetime(exp_time, format="%Y-%m-%dT%H_%M_%S")
    else:
        [exp_date, exp_hour] = csv_file_directory.stem.split("events")[1].split("T")
        exp_time = exp_date + " " + exp_hour
        test_dt = pd.to_datetime(exp_time, format="%Y-%m-%d %H_%M_%S")
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
            temperature_pattern = "lux*.csv"
            termperature_pattern_meta="meta*.csv"
            led_files = find_file(thisDir, temperature_pattern,termperature_pattern_meta)
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
                temperature_pattern = "temperature_humidity*.txt"
            if temperature_data_options == "DL220THP":
                temperature_pattern = "DL220*.csv"
            this_temp_directory = find_file(thisDir, temperature_pattern)
            try:
                tem_df = load_temperature_data(txt_path=this_temp_directory)
                start_time_tem = find_nearest(tem_df.index, test_dt)
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

    ##here to start analying stimulus-evoked response.
    ## if the tracking data is logged on bonsai (come from fictrac realtime). The file name is tracking_events...
    if tracking_csv_file_directory != None and fictrac_posthoc_analysis is False:
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

    stim_info = pd.read_csv(csv_file_directory)
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
        if (
            visual_paradigm_name == "gratings"
            or visual_paradigm_name == "sweeploom"
            or visual_paradigm_name == "conflict"
            or visual_paradigm_name == "looming"
            or visual_paradigm_name == "sweeping" 
        ):
            if stim_info['PreMovDuration'].unique()!=0:
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
            else:
                stim_info["stim_onset"] = stim_onset
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
    elif df["timestamp"][1] != df["timestamp"][2]:
        # compare the timestamp between event file and dat file
        data_timestamp = df.loc[:, "timestamp"]
        stimulus_timestamp_b = np.tile(stimulus_timestamp, (len(data_timestamp), 1))
        data_timestamp_b = np.tile(data_timestamp, (len(stimulus_timestamp), 1))
        timestamp_diff = data_timestamp_b > np.transpose(stimulus_timestamp_b)
        stim_onset = np.where(np.roll(timestamp_diff, 1, axis=1) != timestamp_diff)[1][
            1::2
        ]
        stim_info["stim_onset"] = stim_onset
    else:
        # compare the timestamp between event file and dat file
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
        if (
            visual_paradigm_name == "gratings"
            or visual_paradigm_name == "sweeploom"
            or visual_paradigm_name.endswith("sities")
            or visual_paradigm_name == "conflict"
            or visual_paradigm_name == "looming"
            or visual_paradigm_name == "sweeping" 
        ):
            stim_to_analyse = stim_info["stim_onset"]
            if type(analysis_methods.get("interval_duration")) == list and np.unique(stim_info["ISI"].values)>0:
                ISI_duration = stim_info["ISI"].values
            elif np.unique(stim_info["ISI"].values)<0:
                ISI_duration=np.round((stim_appearance_onset[1:]-isi_onset[:-1])/camera_fps)
                ISI_duration=np.append(ISI_duration, np.nan)
            else:
                ISI_duration = np.repeat(
                    analysis_methods.get("interval_duration"), stim_to_analyse.shape[0]
                )
            preStim_ISI_duration = np.append(
                analysis_methods.get("prestim_duration"), ISI_duration[:-1]
            )
            num_stim=stim_info.shape[0]
        else:
            stim_to_analyse = stim_info.loc[1::2, "stim_onset"]
            if log_stim_ISI_duration:
                preStim_ISI_duration = np.array(
                    stim_info["Duration"][stim_info.loc[::2, "stim_onset"].index]
                )
            else:
                preStim_ISI_duration=stim_info
            num_stim=int(stim_info.shape[0]/2)
    stim_name_arr = np.array(stim_info["stim_type"][stim_to_analyse.index])
    present_duration_arr = np.array(stim_info["Duration"][stim_to_analyse.index])
    (
        summary_df,
        cum_travel_distance,
        velocity_arr,
        yaw_rotation_tbt,
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

    cum_yaw_rotation = np.nancumsum(yaw_rotation_tbt, axis=1)
    # add trial info to summary_df
    summary_df["stim_type"] = stim_name_arr
    summary_df["present_trial_duration"] = present_duration_arr
    summary_df["preStim_ISI_duration"] = preStim_ISI_duration

    print("ready to plot graph with some data and decompose those lists")
    COL = MplColorHelper(colormap_name, 0, len(stim_type))
    if analysis_methods.get("plotting_tbt_overview") == True:
        if visual_paradigm_name=='coherence':## this should depends on the lengh of analysis window prior to stimulus
            running_trial_distance_threshold=10
        else:
            running_trial_distance_threshold=10
        fig1 = plt.figure(figsize=(18, 7), tight_layout=True)
        ax = plt.subplot(241)
        if "opto_index" in summary_df.columns:
            ax.hist(summary_df["opto_index"].values, bins=20)
        # ax.set_xlabel("optomotor index")
        ax.set_ylabel("frequency")
        ax.set_title("optomotor response during stimulus presentation")
        ax1 = plt.subplot(242)
        ax1.hist(summary_df["travel_distance"].values, bins=20)
        # ax1.set_xlabel("travel distance (mm)")
        # ax1.set_ylabel("frequency")
        ax1.set_title("travel distance during stimulus presentation")
        ax3 = plt.subplot(245)
        if "opto_index_isi" in summary_df.columns:
            ax3.hist(summary_df["opto_index_isi"].values, bins=20)
        ax3.set_xlabel("optomotor index")
        ax3.set_ylabel("frequency")
        ax3.set_title("optomotor response 5 sec prior to the stimulus")
        ax4 = plt.subplot(246)
        ax4.hist(summary_df["travel_distance_isi"].values, bins=20)
        ax4.set_xlabel("travel distance (mm)")
        # ax4.set_ylabel("frequency")
        ax4.set_title("travel distance 5 sec prior to the stimulus")
        
        ax_stationary = plt.subplot(243)
        ax_runnning = plt.subplot(247)
        for i in range(len(stim_type)):
            this_coherence = stim_type[i]
            if type(stim_duration)==list:
                these_velocities = velocity_arr[(summary_df["stim_type"] == this_coherence) & (summary_df['present_trial_duration']>duration_for_optomotor_index),:]
                these_travel_distance_isi=summary_df["travel_distance_isi"][(summary_df["stim_type"] == this_coherence) & (summary_df['present_trial_duration']>duration_for_optomotor_index)].values
            else:
                these_velocities = velocity_arr[summary_df["stim_type"] == this_coherence, :]
                these_travel_distance_isi=summary_df["travel_distance_isi"][summary_df["stim_type"] == this_coherence, :].values
            
            stationary_trial=these_velocities[these_travel_distance_isi<running_trial_distance_threshold]
            time_points = np.arange(these_velocities.shape[1])
            if stationary_trial.shape[0]>0:
                time_points_b = np.tile(time_points, (stationary_trial.shape[0], 1))
                ax_stationary.scatter(time_points_b, stationary_trial, s=1, color=COL.get_rgb(i),label=str(this_coherence))
                #ax_stationary.plot(time_points_b, stationary_trial, color=COL.get_rgb(i),label=str(this_coherence))
            running_trial=these_velocities[these_travel_distance_isi>running_trial_distance_threshold]
            if running_trial.shape[0]>0:
                time_points_b = np.tile(time_points, (running_trial.shape[0], 1))
                ax_runnning.scatter(time_points_b, running_trial, s=1, color=COL.get_rgb(i),label=str(this_coherence))
                #ax_runnning.plot(time_points_b, running_trial, color=COL.get_rgb(i),label=str(this_coherence))
        ax_runnning.legend()
        ax_stationary.legend()
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
            velocity_arr = tmp[0, :, :]
            yticks = np.arange(int(num_stim/len(stim_type)/2),num_stim,int(num_stim/len(stim_type)))
            if type(stim_type[0])==str:
                yticklabels = [this_stim_type for this_stim_type in stim_type]
            else:
                yticklabels = [str(this_stim_type) for this_stim_type in stim_type]
            ax2.set(
                yticks=yticks,
                yticklabels=yticklabels,
            )
            for this_y_mark in np.arange(int(num_stim/len(stim_type)),num_stim,int(num_stim/len(stim_type))):
                ax2.axhline(y=this_y_mark,xmin=0,xmax=velocity_arr.shape[1],linewidth=0.3,color='w')
            #trans=mt.blended_transform_factory(ax2.transData,ax2.transAxes)
            #ax2.pylab.gca()
            #ax2.hlines(y=np.arange(int(num_stim/len(stim_type)),num_stim,int(num_stim/len(stim_type))),xmin=0,xmax=velocity_arr.shape[1],linewidth=10,color='w',transform=trans)
        else:
            ax2.set_yticks([])
            ax2.set_ylabel("Trial#")

        pcm = ax2.pcolormesh(velocity_arr, cmap="magma", vmin=0, vmax=100)
        # pcm=ax2.pcolormesh(velocity_tbt, cmap = 'magma') ##sometimes, we dont need a vmax
        ## add colorbar
        fig1.colorbar(pcm)

        ## add vertical lines to indicate stimulus onset and offset
        ax2.axvline(
            x=init_frame,
            color="white",
            linestyle="--",
            label="Stimulus Starts",
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
        plt.show()
    if analysis_methods.get("plotting_position_dependant_fixation") == True:

        colormap = np.array(["k", "r", "b", "g", "y", "m", "c", "k"])
        fig5, (ax1, ax2, ax3) = plt.subplots(
            nrows=1, ncols=3, figsize=(18, 7), tight_layout=True
        )
        f_count = 0
        s_count = 0
        m_count = 0
        for i in range(len(stim_type)):
            this_type_stim = stim_type[i]
            this_response = np.diff(
                cum_yaw_rotation[summary_df["stim_type"] == this_type_stim, :]
            )
            time_points = np.arange(this_response.shape[1])
            time_points_b = np.tile(time_points, (this_response.shape[0], 1))

            if this_type_stim.endswith("fast"):
                ax1.plot(
                    np.mean(time_points_b, axis=0),
                    np.mean(this_response, axis=0),
                    color=colormap[f_count],
                    alpha=0.2,
                )
                f_count += 1
            elif this_type_stim.endswith("slow"):
                ax3.plot(
                    np.mean(time_points_b, axis=0),
                    np.mean(this_response, axis=0),
                    color=colormap[s_count],
                    alpha=0.2,
                )
                s_count += 1
            else:
                ax2.plot(
                    np.mean(time_points_b, axis=0),
                    np.mean(this_response, axis=0),
                    color=colormap[m_count],
                    alpha=0.2,
                )
                m_count += 1
        ax1.axvline(
            x=init_frame,
            color="black",
            linestyle="--",
            label="Stimulus Starts",
        )
        ax1.axvline(
            x=init_frame + camera_fps * stim_duration[0],
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
            x=init_frame + camera_fps * stim_duration[1],
            color="black",
            linestyle="--",
            label="Stimulus Ends",
        )
        ax3.axvline(
            x=init_frame,
            color="black",
            linestyle="--",
            label="Stimulus Starts",
        )
        ax3.axvline(
            x=init_frame + camera_fps * stim_duration[2],
            color="black",
            linestyle="--",
            label="Stimulus Ends",
        )
        plot_name = f"conflict_analysis_overview.png"
        if save_output == True:
            fig5.savefig(thisDir / plot_name)
        plt.show()

    if analysis_methods.get("plotting_optomotor_response") == True:
        # colormap = np.array(["k", "y", "m", "c", "r", "b", "g", "k"])
        fig2, (ax3, ax4) = plt.subplots(
            nrows=1, ncols=2, figsize=(36, 14), tight_layout=True
        )
        for i in range(len(stim_type)):
            this_coherence = stim_type[i]
            if type(stim_duration)==list:
                this_response = summary_df.loc[(summary_df["stim_type"] == this_coherence) & (summary_df['present_trial_duration']>duration_for_optomotor_index),"opto_index"].values
            else:
                this_response = summary_df.loc[
                    summary_df["stim_type"] == this_coherence, "opto_index"
                ].values
            if len(this_response) == 0:
                continue
            mean_response = np.nanmean(this_response, axis=0)
            sem_response = np.nanstd(this_response, axis=0, ddof=1) / np.sqrt(
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
            if type(stim_duration)==list:
                this_rotational_velocity = cum_yaw_rotation[(summary_df["stim_type"] == this_coherence) & (summary_df['present_trial_duration']>duration_for_optomotor_index),:]
            else:
                this_rotational_velocity = cum_yaw_rotation[summary_df["stim_type"] == this_coherence, :]
            time_points = np.arange(this_rotational_velocity.shape[1])
            time_points_b = np.tile(time_points, (this_rotational_velocity.shape[0], 1))
            # ax4.scatter(time_points_b, this_rotational_velocity, s=1, color=custom_cmap[i])
            ax4.scatter(time_points_b, this_rotational_velocity, s=1, color=COL.get_rgb(i))

        ax3.set_ylim(-1, 1)
        # xticks=stim_type,
        ax3.set(
            xlabel=visual_paradigm_name,
            yticks=[-1, 0, 1],
            ylabel="Optomotor Index during stimulus",
        )
        # xticks=[np.max(stim_type), 0, np.min(stim_type)], np.sort(summary_df.stim_type.unique())
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

        yaw_orientation_limit = 200
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
            yticks=[y_axis_lim[0], 0, y_axis_lim[1]],
            ylim=[y_axis_lim[0], y_axis_lim[1]],
            ylabel="Orientation (rad)",
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
        plt.show()

    if analysis_methods.get("plotting_deceleration_accerleration") == True:
        travel_distance_after_stim_presence = cum_travel_distance[
            :, init_frame + presence_duration * camera_fps
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
            cum_travel_distance[running_trial, init_frame + 1 * camera_fps]
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
            trial_to_analyse, init_frame, camera_fps, presence_duration
        )
        dec_trial_id = detected_bool[0]
        # this reverse the frame number back so the output is the frame number no. at the detection window
        de_test = camera_fps * presence_duration - detected_bool[1]
        ##remove trials that the true value is outside this 1-sec window
        dev_timing = de_test[de_test != camera_fps * presence_duration]
        dev_trial_id = dec_trial_id[de_test != camera_fps * presence_duration]
        dev_timing_from_analysis_window = (
            dev_timing
            + init_frame
            + (0.5 * presence_duration + abs(analysis_window[0])) * camera_fps
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

    if save_output == True and overwrite_curated_dataset == True:
        summary_file_name = thisDir / "behavioural_summary.pickle"
        summary_df.to_pickle(summary_file_name)
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
            np.save(f, yaw_rotation_tbt)
        print("Save velocity, angular velocity into array and behavioural summary as dataframe. Move to the next animal")
        json_string = json.dumps(analysis_methods, indent=1)
        with open(thisDir / "analysis_methods_dictionary_backup.json", "w") as f:
            f.write(json_string)

    elif overwrite_curated_dataset == False:
        print(f"dont overwrite existing database.")
    else:
        print("dont save anything")

    return summary_df, velocity_arr, yaw_rotation_tbt, travel_distance_whole_session


if __name__ == "__main__":
    # weird files 50525355565759606269727679 fixed by threshold 46 47 52 59 53 fixed by filtering trial in different way 4150556272
    # GN24098 seems not follow the dot motion
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24021/240717/coherence/session1"  # good for checking fft, where potential false nagative show up;1:11:33;1:15:08
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24078/240804/coherence/session1"  # good for checking fft, where 2 comes from
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24076/240804/coherence/session1"
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24084/240805/coherence/session1"# good for checking fft, where false nagative show up
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24114\240811\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24098\240810\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24089\240810\coherence\session1"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24064\240803\coherence\session1"
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
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25102\250929\looming\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25101\250917\looming\session1"
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25102\250929\coherence\session1"
    #thisDir = r"Y:\GN25051\251101\looming\session2"# locust loom prestim last 60 sec-only #non-walking session
    #thisDir = r"Y:\GN25053\251108\looming\session2"# locust loom prestim last 60 sec-only #non-walking session
    #thisDir = r"Y:\GN25053\251108\looming\session6"# locust loom prestim last 60 sec-only walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25058\251123\looming\session3"
    #thisDir = r"Y:\GN25060\251130\sweeping\session1"
    #thisDir = r"Y:\GN25059\251129\sweeping\session1"
    #thisDir = r"Y:\GN25057\251122\sweeping\session1"
    #thisDir = r"Y:\GN25053\251108\looming\session5"# sequential looming, so the prestim last 60 sec walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25060\251130\looming\session2"# sequential looming, so the prestim last 60 sec walking session, did not walk much
    #thisDir = r"Y:\GN25051\251101\looming\session1"# walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25053\251108\looming\session1"# walking session, just one or two jumping epochs
    #thisDir = r"Y:\GN25053\251108\looming\session7"# walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25052\251102\looming\session1"# walking session, also showing jumping epochs
    #thisDir = r"Y:\GN25060\251130\looming\session1"
    #thisDir = r"Y:\GN25060\251130\looming\session4"
    #thisDir = r"Y:\GN25059\251129\looming\session1"#walking session
    #thisDir = r"Y:\GN25058\251123\looming\session2"
    #thisDir = r"Y:\GN25057\251122\looming\session2"
    #thisDir = r"Y:\GN25059\251129\looming\session1"
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
    thisDir = r"Y:\GN25059\251129\coherence\session1"#horizontal coherence
    #thisDir = r"Y:\GN25060\251130\coherence\session1"
    #thisDir = r"Y:\GN25052\251102\coherence\session1"#had issue detecting transition onset. prestim lasts 180 sec only. Animal did not move much.

    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN25102\250929\sweeping\session1"
    # json_file = r"C:\Users\neuroPC\Documents\GitHub\UnityDataAnalysis\analysis_methods_dictionary.json"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
