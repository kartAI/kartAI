import json
import os

import env
from kartai.datamodels_and_services.DatasetBuilder import DatasetBuilder
from kartai.tools.create_training_data import (DatasetBuilder, Region,
                                               getImageSets, getTileGrid, getImageSources)
from kartai.datamodels_and_services.ImageSourceServices import Tile


def fetch_data_to_predict(geom, config_path):
    with open(config_path) as f:
        config = json.load(f)

    tile_grid = getTileGrid(config)
    training_dataset_dir = env.get_env_variable("cached_data_directory")

    image_sources, train_set, valid_set, test_set = getImageSources(
        config, training_dataset_dir, tile_grid)

    image_sets = getImageSets(
        config, image_sources)

    dataset_builder = DatasetBuilder(image_sets)
    if("ProjectArguments" in config):
        dataset = list(dataset_builder.assemble_data(
            Region(geom), config["ImageSources"], project_config=config["ProjectArguments"]))
    else:
        dataset = list(dataset_builder.assemble_data(
            Region(geom), config["ImageSources"]))

    # Save file image references
    data_path = env.get_env_variable('created_datasets_directory')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    with open(data_path + "/prediction_set.json", "w") as file:
        json.dump(Tile.tileset_to_json(dataset), file)
