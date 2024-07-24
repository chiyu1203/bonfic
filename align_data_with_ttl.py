import os, fnmatch, sys, time, json, cv2, warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file, find_file_multiple_patterns


def find_transitions(arr):
    transitions = []
    for i in range(1, len(arr)):
        if arr[i - 1] == False and arr[i] == True:
            transitions.append(i)

    return np.array(transitions)


# Example usage
boolean_array = [False, False, True, False, True, True, False, True, False]
transition_indices = find_transitions(boolean_array)
print("Transition indices:", transition_indices)


def edge_detector(inp):
    return np.arange(len(inp) - 1)[~inp[:-1] & inp[1:]]


def detect_transition(arr, on_threshold, off_threshold, sample_rate=100):
    # pre_stim_interval = 180
    # max_value=np.max(arr[:100])
    # np.min(arr[:18000])
    pass_threshold = arr > on_threshold
    # transition = edge_detector(pass_threshold)
    b2w = find_transitions(pass_threshold)
    # set an arbitary threshold to remove any kind of instrumental noise in the first 30 secs, usually during pre-stim interval
    b2w = b2w[b2w > sample_rate * 30]
    pass_threshold = off_threshold > arr
    w2b = find_transitions(pass_threshold)
    w2b = w2b[w2b > sample_rate * 30]
    return b2w, w2b
    # =arr[transition]>0
    # bright = arr[arr > on_threshold]
    # ttl_diff = np.diff(bright)
    # ttl_off = arr[off_threshold > arr]
    # fig = plt.figure(figsize=(6, 3))
    # ax = fig.add_subplot(111)
    # ax.axis("equal")
    # ax.scatter(np.arange(arr.shape[0]), arr)
    # ax.scatter(np.arange(ttl_diff.shape[0]), ttl_diff)
    # plt.show()


def get_led_value(this_video, analysis_methods):
    # Load a grayscale video frame
    video_capture = cv2.VideoCapture(this_video)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    gray_list = []
    while video_capture.isOpened():
        # Read a frame from the video

        ret, frame = video_capture.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
            # print("Error reading video frame.")
            # exit(1)

        # Define the coordinates of the top-left and bottom-right corners of the ROI
        x, y = 286, 12  # Top-left corner36,20,20,24 286,12,27,33
        width, height = 27, 33  # Width and height of the ROI

        # Extract the ROI from the grayscale frame
        roi = frame[y : y + height, x : x + width]

        # Measure the pixel values within the ROI
        # You can access individual pixel values using numpy indexing
        # for row in roi:
        #     for pixel in row:
        #         # Access the pixel value (gray level)
        #         gray_value = pixel
        #         print(f"Pixel Value (Gray Level): {gray_value}")

        # You can also use numpy functions for operations on the ROI
        # For example, to get the mean gray value of the ROI
        mean_gray = roi.mean()
        gray_list.append(mean_gray)
        print(f"Mean Gray Value: {mean_gray}")

        # Display the ROI
        #     cv2.imshow("Live", gray)

        #     # exiting the loop
        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows()

    return gray_list


def analyse_led_signal(thisDir, json_file, stim_info):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    sample_rate = analysis_methods.get("camera_fps")
    visual_paradigm_name = analysis_methods.get("experiment_name")
    if visual_paradigm_name == "gratings" or visual_paradigm_name == "sloom":
        trial_num = stim_info.shape[0]
    else:
        trial_num = int(stim_info.shape[0] / 2)
    pd_pattern = "PD*.csv"
    this_PD = find_file(thisDir, pd_pattern)

    trial_pattern = "trial*.csv"
    this_trial_info = find_file(thisDir, trial_pattern)
    [exp_date, exp_hour] = this_trial_info.stem.split("events")[1].split("T")
    exp_place = this_trial_info.parts[-6]

    video_pattern = "*.avi"
    this_video = find_file(thisDir, video_pattern)
    # file_name, file_extension = os.path.splitext(file_path)
    # output_file_name = file_name + "_timestamp.csv"

    # Load videos
    led_pattern = ["*sync_pulse.csv", "*LED.csv", "lux*.csv"]
    led_files = find_file_multiple_patterns(thisDir, led_pattern)
    if led_files is None:
        return print("led file is not found")

    fictrac_vidLog_pattern = "fictrac-vidLogFrames*.txt"
    fictrac_vidLog = find_file(thisDir, fictrac_vidLog_pattern)
    led_value = []
    on_thresholds = []
    off_thresholds = []
    """
    if len(led_files) == 0:
        on_thresholds = [25000, 80000]
        off_thresholds = [1000, 1000]
        for i in range(len(this_video)):
            grey_list = get_led_value(this_video[i], analysis_methods)
            led_value.append(grey_list)
    """
    if isinstance(led_files, list):
        print(f"multiple led files {led_files} were found. They came from bonsai")
        std_weight = 100
        for i in range(len(led_files)):
            arr = np.genfromtxt(led_files[i], delimiter=",", dtype=int)
            if arr.ndim == 2:
                arr = arr[:, 0]
            sort_arr = arr[np.argsort(arr)]
            lower_mean = np.mean(sort_arr[10000:20000])
            lower_std = np.std(sort_arr[10000:20000])
            higher_mean = np.mean(sort_arr[-30000:-20000])
            higher_std = np.std(sort_arr[-30000:-20000])
            # if exp_place == "locustMatrex" and visual_paradigm_name == "coherence":
            if exp_place == "MatrexVR":
                on_thresholds.append(int(np.mean(sort_arr)))
                off_thresholds.append(int(np.mean(sort_arr)))
            # elif exp_place == "locustMatrex" and visual_paradigm_name == "gratings":
            #     std_weight = 40
            #     on_thresholds.append(int(higher_mean - higher_std * std_weight))
            #     off_thresholds.append(int(lower_mean + lower_std * std_weight))
            else:
                on_thresholds.append(int(higher_mean - higher_std * std_weight))
                off_thresholds.append(int(lower_mean + lower_std * std_weight))
            led_value.append(arr)
    elif len(led_files.stem) > 0:
        print(f"using led file {led_files}, extracted from bonsai or elsewhere")
        arr = np.genfromtxt(led_files, delimiter=",", dtype=int)
        sort_arr = arr[np.argsort(arr)]
        lower_mean = np.mean(sort_arr[:10000])
        lower_std = np.std(sort_arr[:10000])
        higher_mean = np.mean(sort_arr[-10000:])
        higher_std = np.std(sort_arr[-10000:])
        # on_thresholds = [17000]
        # off_thresholds = [6000]
        on_thresholds.append(int(higher_mean - higher_std * 100))
        off_thresholds.append(int(lower_mean + lower_std * 100))
        led_value.append(arr)
    else:
        print(f"LED file not found. Use openCV to analyse the LED signals")
        on_thresholds = [25000, 80000]
        off_thresholds = [1000, 1000]
        for i in range(len(this_video)):
            grey_list = get_led_value(this_video[i], analysis_methods)
            led_value.append(grey_list)

    ## use led value to detect the transition.
    for i in range(len(led_value)):
        b2w, w2b = detect_transition(
            led_value[i], on_thresholds[i], off_thresholds[i], sample_rate
        )
        """hardcode here for the pilot experiment due to sync path for 0 coherence stimulus is programmed into the wrong colour"""
        if exp_date == "2024-07-19" or exp_date == "2024-07-17":
            duration_med = np.median(np.diff(b2w))
            num_stim_inbetween = np.round(np.diff(b2w) / duration_med)
            missing_onset = np.where(num_stim_inbetween.astype(int) >= 2)
            missing_onset_num = num_stim_inbetween.astype(int)[missing_onset]
            for i, j in zip(missing_onset[0][::-1], missing_onset_num[::-1]):
                if j == 2:
                    b2w = np.insert(b2w, i + 1, b2w[i] + duration_med)
                    w2b = np.insert(w2b, i, w2b[i - 1] + duration_med)
                if j == 3:
                    b2w = np.insert(b2w, i + 1, b2w[i] + duration_med)
                    b2w = np.insert(b2w, i + 2, b2w[i] + duration_med * 2)
                    w2b = np.insert(w2b, i, w2b[i - 1] + duration_med)
                    w2b = np.insert(w2b, i + 1, w2b[i - 1] + duration_med * 2)

        b2w = b2w[: trial_num + 1]
        w2b = w2b[:trial_num]

    ## this probably means the same as dict fictrac_posthoc_analysis, if fictrac_vidLog exists, then probably do not posthoc analysis
    if fictrac_vidLog:
        arr = np.genfromtxt(fictrac_vidLog, delimiter=",", dtype=int)
        stim_onset_frame = arr[w2b]
        isi_onset_frame = arr[b2w]
    elif visual_paradigm_name == "gratings":
        stim_onset_frame = b2w[: len(w2b)]
        isi_onset_frame = w2b
    else:
        stim_onset_frame = w2b
        isi_onset_frame = b2w[: len(w2b)]
    return stim_onset_frame, isi_onset_frame


if __name__ == "__main__":
    # thisDir = "C:/Users/neuroLaptop/Documents/Open Ephys/2023-06-06_21-51-30/"
    # thisDir = r"Z:\Users\chiyu\sync_test"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23001\231121\coherence\session1"
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23001\231121\coherence\session2"
    json_file = "analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    analyse_led_signal(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
