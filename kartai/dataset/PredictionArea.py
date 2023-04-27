import json
import os

import env
from kartai.datamodels_and_services.DatasetBuilder import DatasetBuilder
from kartai.tools.create_training_data import (DatasetBuilder, Region,
                                               getImageSets, getTileGrid, get_image_sources)
from kartai.datamodels_and_services.ImageSourceServices import Tile


def fetch_data_to_predict(geom, config_path, output_path, start_iteration=None):
    with open(config_path) as f:
        config = json.load(f)

    tile_grid = getTileGrid(config)
    training_dataset_dir = env.get_env_variable("cached_data_directory")

    image_sources, train_set, valid_set, test_set = get_image_sources(
        config, training_dataset_dir, tile_grid, eager_load=True, start_iteration=start_iteration)

    image_sets = getImageSets(
        config, image_sources)

    dataset_builder = DatasetBuilder(image_sets)
    if("ProjectArguments" in config):
        dataset = list(dataset_builder.assemble_data(
            Region(geom), config["ImageSources"], project_config=config["ProjectArguments"], eager_load=True, start_iteration=start_iteration))
    else:
        dataset = list(dataset_builder.assemble_data(
            Region(geom), config["ImageSources"], eager_load=True, start_iteration=start_iteration))

    # Save file image references
    with open(output_path, "w") as file:
        json.dump(Tile.tileset_to_json(dataset), file)
