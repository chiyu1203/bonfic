import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "exp_place": "VCCball",
    "experiment_name": "densities",
    "temperature_data_options": None,
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "debug_mode": False,
    "fictrac_posthoc_analysis": False,
    "use_led_to_align_stimulus_timing": False,
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
    "optomotor_index_duration": 25,
    "camera_fps": 144,
    "trackball_radius": 40,
    "monitor_fps": 144,
    "stim_duration": 25,
    "interval_duration": 5,
    "prestim_duration": 60,
    "stim_type": [
        -4096,
        -1024,
        -256,
        -64,
        64,
        256,
        1024,
        4096,
    ],
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
