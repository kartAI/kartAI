import geopandas as gp
from glob import glob
import env
import os

from kartai.utils.crs_utils import get_defined_crs_from_config


def get_geo_data_frame(files, crs):
    gdf = gp.GeoDataFrame()
    for i in range(len(files)):
        gdf = gdf.append(
            gp.read_file(files[i]), ignore_index=True)

    gdf.crs = crs
    return gdf


def get_IoU(label_files, prediction_gdf, area_to_predict, crs):
    label_gdf = get_geo_data_frame(label_files, crs)

    aoi_gdf = gp.read_file(area_to_predict)
    aoi_gdf.crs = crs

    label_gdf = gp.clip(label_gdf, aoi_gdf)
    prediction_gdf = gp.clip(prediction_gdf, aoi_gdf)

    intersection = gp.overlay(label_gdf, prediction_gdf,
                              how='intersection', keep_geom_type=False).geometry.area.sum()
    label_area = label_gdf.geometry.area.sum()
    prediction_area = prediction_gdf.geometry.area.sum()

    iou = intersection / (label_area + prediction_area - intersection)

    return iou

CRS_kristiansand = 'EPSG:25832'

def get_IoU_for_ksand(prediction_gdf):

    label_bygning_path = os.path.join(env.get_env_variable(
        'cached_data_directory'), "AP2_T2_geodata/Prosjektområde/shape/bygning.shp")
    label_tiltak_path = os.path.join(env.get_env_variable(
        'cached_data_directory'), "AP2_T2_geodata/Prosjektområde/shape/tiltak.shp")
    label_files = [label_bygning_path, label_tiltak_path]

    area_to_predict = os.path.join(env.get_env_variable(
        'cached_data_directory'), "AP2_T2_geodata/Prosjektområde/shape/avgrensning.shp")

    iou = get_IoU(label_files, prediction_gdf,
                  area_to_predict, crs=CRS_kristiansand)
    print(str(iou))
    return iou
