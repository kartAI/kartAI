from env_secrets import get_env_secret

config = {
    "AZURE_STORAGE_CONNECTION_STRING": get_env_secret('AZURE_STORAGE_CONNECTION_STRING'),
    "AZURE_BYGG_POSTGRESQL_PSW": get_env_secret('AZURE_BYGG_POSTGRESQL_PSW'),
    "NK_WMS_API_KEY": get_env_secret('NK_WMS_API_KEY'),
    "metadata_container_name": "modelsmetadata-v2",
    "models_container_name": "models-v2",
    "ksand_performances_container_name": "ksand-performances",
    "building_datasets_container_name": "building-datasets",
    "trained_models_directory": "checkpoints",
    "prediction_results_directory": "results",
    "cached_data_directory": 'training_data',
    "created_datasets_directory": 'training_data/created_datasets/'
}


def get_env_variable(variable):
    return config[variable]
