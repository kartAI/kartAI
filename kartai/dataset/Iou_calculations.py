import geopandas as gp


def get_iou_for_region(prediction_gdf: gp.GeoDataFrame, true_labels: gp.GeoDataFrame, region, crs: str):
    """Get intersection over union from predictions in a region"""
    aoi_gdf = gp.read_file(region)
    aoi_gdf.set_crs(crs, inplace=True, allow_override=True)

    intersection = gp.overlay(true_labels, prediction_gdf,
                              how='intersection', keep_geom_type=False).geometry.area.sum()
    label_area = true_labels.geometry.area.sum()
    prediction_area = prediction_gdf.geometry.area.sum()

    iou = intersection / (label_area + prediction_area - intersection)

    return iou
