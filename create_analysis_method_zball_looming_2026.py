import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "exp_place": "Zball",
    "experiment_name": "looming",
    "temperature_data_options": "trial",
    "overwrite_curated_dataset": True,
    "load_previous_methods": False,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "save_output": True,
    "fictrac_posthoc_analysis": True,
    "use_led_to_align_stimulus_timing": True,
    "align_with_isi_onset": False,
    "mark_jump_as_nan": False,
    "active_trials_only": True,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "plotting_trajectory": True,
    "plotting_event_related_trajectory": True,
    "plotting_deceleration_accerleration": False,
    "plotting_position_dependant_fixation": False,
    "plotting_optomotor_response": False,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": False,
    "analysis_by_stimulus_type": True,
    "duration_for_optomotor_index": 0,
    "yaw_axis":"z",
    "camera_fps": 100,
    "trackball_radius": 5,
    "monitor_fps": 144,
    "prestim_duration": 60,
    "stim_duration": [1,8],
    "interval_duration": [10,15,20],
    "analysis_window": [
        -2,
        9
    ]
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
