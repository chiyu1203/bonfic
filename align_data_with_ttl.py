import os, fnmatch
import time
import json
import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt


def find_transitions(arr):
    transitions = []
    for i in range(1, len(arr)):
        if arr[i - 1] == False and arr[i] == True:
            transitions.append(i)
    return transitions


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
        return os.path.join(thisDir, file_check[0])
    else:
        vid_list = []
        for i in range(len(file_check)):
            vid_list.append(os.path.join(thisDir, file_check[i]))
        return vid_list


def detect_transition(this_led, on_threshold, off_threshold):
    sample_rate = 100
    arr = np.genfromtxt(this_led, delimiter=",", dtype=int)
    # max_value=np.max(arr[:100])
    # np.min(arr[:100])
    pass_threshold = arr > on_threshold
    transition = edge_detector(pass_threshold)
    b2w = find_transitions(pass_threshold)
    pass_threshold = off_threshold > arr
    w2b = find_transitions(pass_threshold)
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
        x, y = 36, 20  # Top-left corner36,20,20,24
        width, height = 20, 24  # Width and height of the ROI

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

    video_pattern = "video*.avi"
    this_video = find_file(thisDir, video_pattern)
    grey_value = []
    # Load videos
    led_pattern = "video*.csv"
    this_led = find_file(thisDir, led_pattern)
    if len(this_led) > 0:
        if isinstance(this_video, list):
            for i in range(len(this_video)):
                grey_list = get_led_value(this_video[i], analysis_methods)
                grey_value.append(grey_list)
        elif isinstance(this_video, str):
            grey_list = get_led_value(this_video, analysis_methods)
            grey_value.append(grey_list)
        else:
            warnings.warn("unable to find a video")
    else:
        print(f"using led file {this_led}, extracted from bonsai or elsewhere")

    on_thresholds = [25000, 80000]
    off_thresholds = [1000, 1000]
    if isinstance(this_led, list):
        for i in range(len(this_video)):
            # on_threshold = 25000
            on_threshold = on_thresholds[i]
            off_threshold = off_thresholds[i]
            b2w, w2b = detect_transition(this_led[i], on_threshold, off_threshold)
    elif isinstance(this_led, str):
        b2w, w2b = detect_transition(this_led, on_thresholds[0], off_thresholds[0])
    else:
        warnings.warn("unable to find a video")


if __name__ == "__main__":
    # thisDir = "C:/Users/neuroLaptop/Documents/Open Ephys/2023-06-06_21-51-30/"
    thisDir = r"Z:\Users\chiyu\sync_test"
    json_file = "analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    analyse_led_signal(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
