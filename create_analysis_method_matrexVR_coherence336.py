import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "exp_place": "MatrexVR",
    "experiment_name": "coherence",
    "temperature_data_options": "bonsai",
    "overwrite_curated_dataset": True,
    "load_previous_methods": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "save_output": True,
    "fictrac_posthoc_analysis": True,
    "use_led_to_align_stimulus_timing": True,
    "align_with_isi_onset": False,
    "mark_jump_as_nan": True,
    "active_trials_only": True,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "plotting_trajectory": False,
    "plotting_event_related_trajectory": False,
    "plotting_deceleration_accerleration": False,
    "plotting_position_dependant_fixation": False,
    "plotting_optomotor_response": True,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": True,
    "analysis_by_stimulus_type": False,
    "duration_for_optomotor_index": 15,
    "yaw_axis":"z",
    "camera_fps": 100,
    "trackball_radius": 5,
    "monitor_fps": 60,
    "stim_duration": [
        15,
        35,
    ],
    "interval_duration": [
        40,
        50,
    ],
    "analysis_window": [-10, 40],
    "prestim_duration": 2400,
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
