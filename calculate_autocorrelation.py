import os, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file
from data_cleaning import sorting_trial_info


def load_data(thisDir, json_file, trial_num):
    thisDir = Path(thisDir)
    # Figuring out path and directory
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())

    visual_paradigm_name = analysis_methods.get("experiment_name")
    stim_info = pd.read_csv(thisDir)
    stim_info, stim_type = sorting_trial_info(
        stim_info, visual_paradigm_name=visual_paradigm_name, exp_place="matrexvr"
    )
    if "trial_num" in locals() and trial_num * 2 < stim_info.shape[0]:
        stim_info = stim_info.iloc[: trial_num * 2, :]
        ylim4 = [75, 95]
        ylim5 = [55, 85]
        ylim6 = [15, 35]
    else:
        ylim4 = [150, 190]
        ylim5 = [110, 170]
        ylim6 = [30, 70]
        print(
            "defined trial number is bigger than total number of trial. So use the entire info."
        )
    lag_length = np.linspace(1, 5, 5, dtype=int)
    lag_data = np.ones((3, lag_length.shape[0]))
    for i in range(lag_length.shape[0]):
        this_lag = lag_length[i] + 1
        lag_data[0, i] = stim_info["ID"][::2].autocorr(lag=this_lag)
        lag_data[1, i] = stim_info["ID"][1::2].autocorr(lag=this_lag)
        lag_data[2, i] = stim_info["stim_type"][1::2].autocorr(lag=this_lag)
    direction = stim_info["stim_type"][1::2] / stim_info["Coherence"][1::2]
    stim_direction_count = direction.value_counts()
    ISI_type_count = stim_info["Duration"][::2].value_counts()
    stim_type_count = stim_info["stim_type"][1::2].value_counts()
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(18, 7), tight_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    ax1.scatter(lag_length, lag_data[0, :])
    ax2.scatter(lag_length, lag_data[1, :])
    ax3.scatter(lag_length, lag_data[2, :])
    ax1.set(title="isi ID autocorrelation", ylim=[-0.25, 0.25])
    ax2.set(title="stim ID autocorrelation", ylim=[-0.25, 0.25])
    ax3.set(title="stim type autocorrelation", ylim=[-0.25, 0.25])
    ax4.scatter(ISI_type_count.index.values, ISI_type_count.values)
    ax4.set(ylim=ylim4, title="ISI type count")
    ax5.scatter(stim_direction_count.index.values, stim_direction_count.values)
    ax5.set(ylim=ylim5, title="stim direction count")
    ax6.scatter(stim_type_count.index.values, stim_type_count.values)
    ax6.set(ylim=ylim6, title="stim type count")
    if "trial_num" in locals():
        fig_name = f"{thisDir.stem}_{trial_num}.jpg"
    else:
        fig_name = f"{thisDir.stem}.jpg"
    fig.savefig(fig_name)


if __name__ == "__main__":
    thisDir = "Z:/Users/chiyu/trial_events44.csv"
    json_file = "./analysis_methods_dictionary.json"
    trial_num = 168
    tic = time.perf_counter()
    load_data(thisDir, json_file, trial_num)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
