import json
import env
from kartai.tools.create_training_data import (DatasetBuilder, Region,
                                               getImageSets, getTileGrid, getImageSources)
from kartai.datamodels_and_services.ImageSourceServices import Tile
from osgeo import ogr


def fetch_data_to_predict(geom: ogr.Geometry, config, output_path, num_processes=None):
    """Download data to run prediction on"""
    tile_grid = getTileGrid(config)
    training_dataset_dir = env.get_env_variable("cached_data_directory")

    image_sources, train_set, valid_set, test_set = getImageSources(
        config, training_dataset_dir, tile_grid, eager_load=True)

    image_sets = getImageSets(
        config, image_sources)

    project_config = config["ProjectArguments"] if "ProjectArguments" in config else None
    dataset_builder = DatasetBuilder(image_sets, config["ImageSources"], project_config=project_config,
                                     eager_load=True, num_processes=num_processes)
    dataset = list(dataset_builder.assemble_data(Region(geom)))

    # Save file image references
    with open(output_path, "w") as file:
        json.dump(Tile.tileset_to_json(dataset), file)
