import json
file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "experiment_name": "dots",
    "update_dataset": True,
    "overwrite_curated_dataset": True,
    "plot_trace": False,
    "debug_mode": False,
    "filtering_method": "sg_filter",
    "load_experiment_condition": True,
    "condition_of_interest": "single",
    "analye_stimulus_by_time" : True,
    "plotting_tbt_overview": False,
    "plotting_trajectory" : False,
    "plotting_event_related_trajectory" : False,
}
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)