import os, fnmatch
import json
import time
import pandas as pd


def create_curated_dataset(myList, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    # csv_pattern='trial_*.csv'
    dat_pattern = "*.dat"
    track_ball_radius = 40
    frame_rate = 144
    # stim_duration = 25 # in sec
    # grey_duration = 5
    for thisDir in myList:
        # list up the files in the dir
        file = os.listdir(thisDir)
        # look for files that matches the pattern
        dat_file = fnmatch.filter(file, dat_pattern)
        dat_file_directory = os.path.join(thisDir, dat_file[0])

        ## load dat file

        raw_data = pd.read_table(dat_file_directory, sep="\s+")
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
        for ind in range(0, raw_data.shape[1]):
            raw_data.iloc[:, ind] = pd.to_numeric(raw_data.iloc[:, ind]).astype(
                "float32"
            )

        ## adjust the unit of the x, y position
        raw_data.iloc[:, 6] = raw_data.iloc[:, 6] * track_ball_radius
        raw_data.iloc[:, 7] = raw_data.iloc[:, 7] * track_ball_radius
        raw_data.iloc[:, 8] = raw_data.iloc[:, 8] * track_ball_radius * frame_rate
        ## adjust the unit of the z vector based on the target frame rate of fictrac
        raw_data.iloc[:, 2] = raw_data.iloc[:, 2]

        ### save the curated_database
        if analysis_methods.get("debug_mode") == False:
            database_name = "database_curated.pickle"
            database_directory = os.path.join(thisDir, database_name)
            if (analysis_methods.get("overwrite_curated_dataset") == False) and (
                os.path.isfile == True
            ):
                continue
            else:
                raw_data.to_pickle(database_directory)


if __name__ == "__main__":
    thisDir = "C:/Users/neuroLaptop/Documents/L23038/230824/dots"
    json_file = "analysis_methods_dictionary.json"
    tic = time.perf_counter()
    create_curated_dataset(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
