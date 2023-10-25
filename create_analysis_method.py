import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "experiment_name": "dots",
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "debug_mode": False,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": False,
    "plotting_trajectory": False,
    "plotting_event_related_trajectory": False,
    "plotting_optomotor_response": True,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": True,
}
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
