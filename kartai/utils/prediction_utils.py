

import os
import env


def get_raster_predictions_dir(region_name, checkpoint_name):
    raster_output_dir = os.path.join(env.get_env_variable(
        "prediction_results_directory"), region_name, checkpoint_name, "rasters")
    return raster_output_dir


def get_contour_predictions_dir(region_name, checkpoint_name):
    contour_output_dir = os.path.join(env.get_env_variable(
        "prediction_results_directory"), region_name, checkpoint_name, "contour")
    return contour_output_dir


def get_vector_predictions_dir(region_name, checkpoint_name):
    vector_output_dir = os.path.join(env.get_env_variable(
        "prediction_results_directory"), region_name, checkpoint_name, "vector")
    return vector_output_dir
