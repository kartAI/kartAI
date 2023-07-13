import geopandas as gp

from kartai.dataset.test_area_utils import get_label_files_dir_for_test_region, get_test_region_avgrensning_dir


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


def get_IoU_for_region(prediction_dataset_gdf, region_name, crs):

    label_files = get_label_files_dir_for_test_region(region_name)
    area_to_predict = get_test_region_avgrensning_dir(region_name)

    iou = get_IoU(label_files, prediction_dataset_gdf,
                  area_to_predict, crs=crs)
    print(str(iou))
    return iou
