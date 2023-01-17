from pathlib import Path
from azure import blobstorage
import sys
import os
import env


def check_for_existing_model(checkpoint_name):

    # If running tests, allow to overwrite
    if checkpoint_name == 'test_model' or "test_data_teacher" in checkpoint_name:
        return

    existing_trained_model_names = existing_trained_model_names = get_existing_model_names()
    if checkpoint_name in existing_trained_model_names:
        raise Exception(
            f'\n---ERROR: trained model with name {checkpoint_name} already exists')


def get_dataset_dirs(dataset_name_input):
    created_datasets_dirs = []
    for dataset_name in dataset_name_input:
        created_datasets_dir = os.path.join(env.get_env_variable(
            'created_datasets_directory'), dataset_name)
        created_datasets_dirs.append(created_datasets_dir)
    return created_datasets_dirs


def get_existing_model_names():
    existing_trained_models = blobstorage.getAvailableTrainedModels()
    existing_trained_model_names = [
        Path(modelname).stem for modelname in existing_trained_models]

    # Check for local models as well as azure models

    trained_models_dir = env.get_env_variable('trained_models_directory')

    if os.path.isdir(trained_models_dir):
        for modelname in os.listdir(trained_models_dir):
            modelname = Path(modelname).stem
            if (modelname not in existing_trained_model_names):
                existing_trained_model_names.append(modelname)

    return existing_trained_model_names
