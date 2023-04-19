import glob
import json
import os
import time
import env
from kartai.dataset.create_building_dataset import run_ml_predictions, save_dataset
from kartai.utils.crs_utils import get_defined_crs_from_config

output_prediction_suffix = "prediction_contour"
default_output_predictions_name = f"_{output_prediction_suffix}"

default_output_dir = os.path.join(env.get_env_variable(
    'prediction_results_directory'), 'predictions_contours')


def create_building_contour_dataset(geom, checkpoint_name, area_name, data_config_path, skip_to_postprocess, output_dir=default_output_dir, max_mosaic_batch_size=200, save_to='azure'):

    with open(data_config_path, "r") as config_file:
        config = json.load(config_file)

    if not skip_to_postprocess:

        run_ml_predictions(checkpoint_name, os.path.join(output_dir,area_name),
                           default_output_predictions_name, data_config_path, geom, save_as_contour=True)

        time.sleep(2)  # Wait for complete saving to disk
