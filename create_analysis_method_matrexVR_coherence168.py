import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "exp_place": "MatrexVR",
    "experiment_name": "coherence",
    "temperature_data_path": None,
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "debug_mode": False,
    "fictrac_posthoc_analysis": True,
    "use_led_to_align_stimulus_timing": True,
    "align_with_isi_onset": False,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "plotting_trajectory": True,
    "plotting_event_related_trajectory": False,
    "plotting_deceleration_accerleration": False,
    "plotting_optomotor_response": True,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": True,
    "analysis_by_stimulus_type": False,
    "camera_fps": 100,
    "trackball_radius": 50,
    "monitor_fps": 60,
    "stim_duration": 25,
    "interval_duration": [
        10,
        15,
    ],
    "prestim_duration": 300,
    "stim_type": [
        -100,
        -66,
        -33,
        0,
        33,
        66,
        100,
    ],
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
