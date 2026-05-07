import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "exp_place": "VCCball",
    "experiment_name": "sweeploom",
    "temperature_data_options": None,
    "overwrite_curated_dataset": True,
    "load_previous_methods": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "save_output": True,
    "fictrac_posthoc_analysis": False,
    "use_led_to_align_stimulus_timing": False,
    "align_with_isi_onset": False,
    "mark_jump_as_nan": False,
    "active_trials_only": True,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "plotting_trajectory": True,
    "plotting_event_related_trajectory": False,
    "plotting_deceleration_accerleration": True,
    "plotting_position_dependant_fixation": False,
    "plotting_optomotor_response": False,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": True,
    "analysis_by_stimulus_type": True,
    "duration_for_optomotor_index":0,
    "yaw_axis":"z",
    "camera_fps": 144,
    "trackball_radius": 4,
    "monitor_fps": 144,
    "stim_duration": [3,4],
    "interval_duration": [5,10,15,20,25],
    "prestim_duration": 60,
    "analysis_window": [
        -1,
        5
    ],
    "stim_type": [
        "c2b_f_sloom",
        "c2b_s_sloom",
        "c2f_f_sloom",
        "c2f_s_sloom",
        "c2c_f_loom",
        "c2c_s_loom",
    ],
}  # plus value representing clockwise, counterclockwise is minus, then the rest is coherence levels
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
