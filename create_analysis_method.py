import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "experiment_name": "dots",
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "debug_mode": True,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "plotting_trajectory": False,
    "plotting_event_related_trajectory": False,
    "plotting_optomotor_response": True,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": True,
    "analysis_by_stimulus_type": True,
    "frame_rate": 144,
    "stim_duration": 25,
    "interval_duration": 5,
    "stim_type": [
        -100,
        -75,
        -50,
        -25,
        0,
        25,
        50,
        75,
        100,
    ],
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
