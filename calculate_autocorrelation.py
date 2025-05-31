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


def load_data(thisDir, json_file, partial_analysis="all"):
    thisDir = Path(thisDir)
    # Figuring out path and directory
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            # print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())

    plotting_tbt_overview = analysis_methods.get("plotting_tbt_overview")
    analysis_methods.update({"experiment_name": "coherence"})
    analysis_methods.update({"exp_place": "Zball"})
    save_output = analysis_methods.get("save_output")
    stim_info = pd.read_csv(thisDir)
    stim_info, stim_type = sorting_trial_info(
        stim_info, analysis_methods)
    stim_info["time_group"] = pd.cut(
        stim_info.index, bins=3, labels=["early", "middle", "late"]
    )
    if partial_analysis == "all":
        ylim4 = [round(stim_info.shape[0] / 2) - 20, round(stim_info.shape[0] / 2) + 20]
        #ylim5 = [round(stim_info.shape[0] / 2) - 20, round(stim_info.shape[0] / 2) + 20]
        ylim6 = [round(stim_info.shape[0] / 7) - 15, round(stim_info.shape[0] / 7) + 15]
        # ylim4 = [round(stim_info.shape[0] / 2) - 20, round(stim_info.shape[0] / 2) + 20]
        # ylim5 = [round(stim_info.shape[0] / 2) - 20, round(stim_info.shape[0] / 2) + 20]
        # ylim6 = [round(stim_info.shape[0] / 7) - 15, round(stim_info.shape[0] / 7) + 15]
    else:
        stim_info = stim_info[stim_info["time_group"] == partial_analysis]
        # ylim4 = [round(stim_info.shape[0] / 6) - 20, round(stim_info.shape[0] / 6) + 20]
        # ylim5 = [round(stim_info.shape[0] / 6) - 20, round(stim_info.shape[0] / 6) + 20]
        # ylim6 = [0, round(stim_info.shape[0] / 21) + 10]

    lag_length = np.linspace(1, 5, 5, dtype=int)
    lag_data = np.ones((3, lag_length.shape[0]))
    for i in range(lag_length.shape[0]):
        this_lag = lag_length[i] + 1
        lag_data[0, i] = stim_info["ID"][::2].autocorr(lag=this_lag)
        lag_data[1, i] = stim_info["ID"][1::2].autocorr(lag=this_lag)
        lag_data[2, i] = stim_info["Coherence"][1::2].autocorr(lag=this_lag)
    direction = stim_info["stim_type"][1::2] / stim_info["Coherence"][1::2]
    stim_direction_count = direction.value_counts()
    ISI_type_count = stim_info["Duration"][::2].value_counts()
    stim_type_count = stim_info["stim_type"][1::2].value_counts()
    if plotting_tbt_overview:
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
        #ax5.set(ylim=ylim5, title="stim direction count")
        ax6.scatter(stim_type_count.index.values, stim_type_count.values)
        ax6.set(ylim=ylim6, title="stim type count")
        if save_output:
            fig_name = f"{thisDir.stem}_{partial_analysis}.jpg"
            fig.savefig(thisDir.parent / fig_name)
    sorted_stim_type = stim_type_count.sort_index().values
    sorted_ISI_type = ISI_type_count.sort_index().values
    sorted_stim_direction = stim_direction_count.sort_index().values
    return sorted_stim_direction, sorted_ISI_type, sorted_stim_type, lag_data


if __name__ == "__main__":
    # thisDir = "Z:/Users/chiyu/trial_events44.csv"#original is 672, meaning 336 trials
    #thisDir = "Z:/DATA/experiment_trackball_Optomotor/MatrexVR/GN24034/240801/coherence/session1/trial_events2024-08-01T13_42_14.csv"
    thisDir = r"Z:\Users\chiyu\random_seed_autocorrelation_analysis\random_3duration_3coherence\trial_events1.csv"
    json_file = "./analysis_methods_dictionary.json"
    partial_analysis = "all"
    # ["early", "middle", "late" or "all"]
    tic = time.perf_counter()
    load_data(thisDir, json_file, partial_analysis)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
