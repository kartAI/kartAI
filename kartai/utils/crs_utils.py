import json
from osgeo import osr


def get_defined_crs_from_config_path(data_config_path: str, only_numbers: bool = False) -> str:
    with open(data_config_path, "r") as config_file:
        config = json.load(config_file)
    return get_defined_crs_from_config(config, only_numbers)


def get_defined_crs_from_config(config: dict, only_numbers: bool = False) -> str:
    tilegrid = config["TileGrid"]
    crs = tilegrid['srid']
    return crs if only_numbers else f'EPSG:{crs}'


def get_projection_from_config_path(data_config_path: str) -> str:
    epsg = get_defined_crs_from_config_path(data_config_path, True)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    projection = srs.ExportToWkt()
    return projection
