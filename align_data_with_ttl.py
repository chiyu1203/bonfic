import os, fnmatch
import time
import json
import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def find_file(thisDir, pattern):
    file_check = fnmatch.filter(os.listdir(thisDir), pattern)
    if len(file_check) == 0:
        print(f"no {pattern} found in {thisDir}. Let's leave this programme")
        return None
    elif len(file_check) == 1:
        # return os.path.join(thisDir, file_check[0])
        return Path(thisDir) / file_check[0]
    else:
        vid_list = []
        for i in range(len(file_check)):
            vid_list.append(Path(thisDir) / file_check[i])
        return vid_list


def detect_transition(arr, on_threshold, off_threshold):
    sample_rate = 100
    trial_num = 112
    # pre_stim_interval = 180
    # max_value=np.max(arr[:100])
    # np.min(arr[:18000])
    pass_threshold = arr > on_threshold
    transition = edge_detector(pass_threshold)
    b2w = find_transitions(pass_threshold)
    # set an arbitary threshold to remove first few noise value
    b2w = b2w[b2w > sample_rate * 60]
    pass_threshold = off_threshold > arr
    w2b = find_transitions(pass_threshold)
    w2b = w2b[w2b > sample_rate * 60]
    return b2w[: trial_num + 1], w2b[:trial_num]
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


def analyse_led_signal(thisDir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())

    pd_pattern = "PD*.csv"
    this_PD = find_file(thisDir, pd_pattern)

    trial_pattern = "trial*.csv"
    this_trial_info = find_file(thisDir, trial_pattern)

    video_pattern = "*.avi"
    this_video = find_file(thisDir, video_pattern)
    # file_name, file_extension = os.path.splitext(file_path)
    # output_file_name = file_name + "_timestamp.csv"

    # Load videos

    led_pattern = "fictrac*.csv"
    led_files = find_file(thisDir, led_pattern)
    fictrac_vidLog_pattern = "fictrac-vidLogFrames*.txt"
    fictrac_vidLog = find_file(thisDir, fictrac_vidLog_pattern)
    led_value = []
    '''
    if len(led_files) == 0:
        on_thresholds = [25000, 80000]
        off_thresholds = [1000, 1000]
        for i in range(len(this_video)):
            grey_list = get_led_value(this_video[i], analysis_methods)
            led_value.append(grey_list)
    '''
    if isinstance(led_files, list):
        on_thresholds = [25000, 80000]
        off_thresholds = [1000, 1000]
        print(f"using led file {led_files}, extracted from bonsai or elsewhere")
        for i in range(len(led_files)):
            arr = np.genfromtxt(led_files[i], delimiter=",", dtype=int)
            led_value.append(arr)
    elif len(led_files.stem)>0:
        on_thresholds = [18000]
        off_thresholds = [5000]
        print(f"using led file {led_files}, extracted from bonsai or elsewhere")
        arr = np.genfromtxt(led_files, delimiter=",", dtype=int)
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
        b2w, w2b = detect_transition(led_value[i], on_thresholds[i], off_thresholds[i])
    ## this probably means the same as dict fictrac_posthoc_analysis, if fictrac_vidLog exists, then probably do not posthoc analysis
    if fictrac_vidLog:
        arr = np.genfromtxt(fictrac_vidLog, delimiter=",", dtype=int)
        stim_onset_frame = arr[w2b]
        isi_onset_frame = arr[b2w]
    else:
        stim_onset_frame = w2b
        isi_onset_frame = b2w[:len(w2b)]
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
