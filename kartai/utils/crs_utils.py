import json


def get_defined_crs_from_config_path(data_config_path):
    with open(data_config_path, "r") as config_file:
        config = json.load(config_file)
    return get_defined_crs_from_config(config)


def get_defined_crs_from_config(config):
    tilegrid = config["TileGrid"]
    crs = tilegrid['srid']
    return f'EPSG:{crs}'
