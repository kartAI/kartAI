import os

config = {
    "NK_WMS_API_KEY": os.environ['NK_WMS_API_KEY'],
    "OSM_DB_PWD": os.environ['OSM_DB_PWD'],
    "metadata_container_name": "modelsmetadata-v2",
    "models_container_name": "models-v2",
    "ksand_performances_container_name": "ksand-performances",
    "balsfjord_performances_container_name": "balsfjord-performances-adjusted",
    "results_datasets_container_name": "building-datasets",
    "trained_models_directory": "checkpoints",
    "prediction_results_directory": "results",
    "cached_data_directory": 'training_data',
    "created_datasets_directory": 'training_data/created_datasets/'
}

def get_env_variable(variable):
    return config[variable]
