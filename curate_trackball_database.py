import os, fnmatch
import re
import csv
import json
import time
import pandas as pd
import chardet
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/neuroPC/Documents/GitHub/behavioral_analysis")
from useful_tools import find_file


def generate_timestamp_csv(file_path):
    save_csv = False
    file_name, file_extension = os.path.splitext(file_path)
    output_file_name = file_name + "_timestamp.csv"
    frame_pattern = r"Trackball::process \[INF\] Frame (\d+)"
    float_pattern = r"BaslerSource::grab \[DBG\] Frame captured \d+x\d+x\d+ @ \d+\.\d+ \((\d+\.\d+)\)"

    # Detect the file's encoding
    with open(file_path, "rb") as rawdata:
        result = chardet.detect(rawdata.read(10000))

    detected_encoding = result["encoding"]

    # Open and read the log file
    with open(file_path, "r", encoding=detected_encoding) as log_file:
        log_content = log_file.read()

    # Use regular expressions to extract frame numbers and float values
    frame_numbers = [int(match) for match in re.findall(frame_pattern, log_content)]
    float_values = [float(value) for value in re.findall(float_pattern, log_content)]
    if save_csv == True:
        # Combine the data into a 2D array
        if len(frame_numbers) == len(float_values):
            data = list(zip(frame_numbers, float_values))
        else:
            frame_numbers.pop()
            data = list(zip(frame_numbers, float_values))

        # Save the extracted data to a new CSV file
        if save_csv == True:
            with open(output_file_name, "w", newline="") as output_csv:
                csv_writer = csv.writer(output_csv)
                csv_writer.writerow(["Frame Number", "Float Value"])
                csv_writer.writerows(data)
                print(f"Data extracted and saved to {output_file_name}")
    else:
        data = float_values

    return data


def create_curated_dataset(thisDir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    # timestamp_pattern='trial_*.csv'
    log_pattern = "*.log"
    dat_pattern = "*.dat"
    track_ball_radius = analysis_methods.get("trackball_radius")
    frame_rate = analysis_methods.get("frame_rate")
    # list up the files in the dir
    pd_pattern = "PD*.csv"
    dat_file = find_file(thisDir, dat_pattern)
    # file = os.listdir(thisDir)
    # look for files that matches the pattern
    # find_file()
    # dat_file = fnmatch.filter(file, dat_pattern)
    # dat_file_directory = os.path.join(thisDir, dat_file[0])

    ## load dat file
    for this_dat_file in dat_file:
        file_name = Path(this_dat_file).stem
        experiment_timestamp = file_name.split("-")
        raw_data = pd.read_table(this_dat_file, sep="\s+")
        ## fictrac title is here 'frame counter', 'delta rotation vector cam x', 'delta rotation vector cam y', \
        ##'delta rotation vector cam z', 'delta rotation error score', 'delta rotation vector lab x', \
        ##'delta rotation vector lab y', 'delta rotation vector lab z', 'absolute rotation vector cam x',\
        ##'absolute rotation vector cam y', 'absolute rotation vector cam z', \
        ##'absolute rotation vector lab x', 'absolute rotation vector lab y', \
        ##'absolute rotation vector lab z', 'intergrated x position','intergrated y position', \
        ##'intergrated animal heading', 'animal movement direction', 'animal movement speed', \
        ##'intergrated forward motion','intergrated side motion', 'timestamp', 'sequence counter','delta timestamp', 'alt. timestamp']

        ## drop some column
        raw_data.drop(
            raw_data.columns[[0, 1, 2, 3, 4, 8, 9, 10, 16, 17, 19, 20, 21, 22, 23]],
            axis=1,
            inplace=True,
        )
        raw_data.columns = [
            "delta rotation vector lab x",
            "delta rotation vector lab y",
            "delta rotation vector lab z",
            "absolute rotation vector lab x",
            "absolute rotation vector lab y",
            "absolute rotation vector lab z",
            "intergrated x position",
            "intergrated y position",
            "instant speed",
            "timestamp",
        ]
        ## cleaning , and change the data from object to float
        raw_data.iloc[:, :] = raw_data.iloc[:, :].replace({",": ""}, regex=True)
        if raw_data["timestamp"][0] == raw_data["timestamp"][1]:
            print(
                "this error timestamp value comes from a bug in new fictrac, extract timestamp from log file"
            )
            f"{file_name}.log"
            this_log_file = f"{file_name}.log"
            timestamp_arr = generate_timestamp_csv(os.path.join(thisDir, this_log_file))
            ## needs to troubleshoot this part to make sure only the timestamp array is used to replace the original part
            if len(timestamp_arr) > raw_data.shape[0]:
                timestamp_arr.pop()
                print(
                    "there is an additional one timestamp in the log file. Remove the last one"
                )

            elif len(timestamp_arr) < raw_data.shape[0]:
                timestamp_arr.append(timestamp_arr[-1])
                print(
                    "missing one timestamp from the log file. repeat the last one again to fill that gap"
                )
            else:
                print("log file size matches with dat file")
            raw_data["timestamp"] = timestamp_arr
        for ind in range(0, raw_data.shape[1]):
            raw_data.iloc[:, ind] = pd.to_numeric(raw_data.iloc[:, ind]).astype(
                "float32"
            )

        ## adjust the unit of the x, y position
        raw_data.iloc[:, 6] = raw_data.iloc[:, 6] * track_ball_radius
        raw_data.iloc[:, 7] = raw_data.iloc[:, 7] * track_ball_radius
        raw_data.iloc[:, 8] = raw_data.iloc[:, 8] * track_ball_radius * frame_rate
        ## adjust the unit of the z vector based on the target frame rate of fictrac
        raw_data.iloc[:, 2] = raw_data.iloc[:, 2] * frame_rate

        ### save the curated_database
        if analysis_methods.get("debug_mode") == False:
            database_name = f"database_curated_{experiment_timestamp[1]}.pickle"
            database_directory = os.path.join(thisDir, database_name)
            if (analysis_methods.get("overwrite_curated_dataset") == False) and (
                os.path.isfile == True
            ):
                print(f"do not overwrite existing pickle file {thisDir}")
            else:
                raw_data.to_pickle(database_directory)


if __name__ == "__main__":
    # thisDir = "Z:/DATA/experiment_trackball_Optomotor/L22146/221106/dots"
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23002\231121\coherence"
    json_file = "analysis_methods_dictionary.json"
    tic = time.perf_counter()
    create_curated_dataset(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
