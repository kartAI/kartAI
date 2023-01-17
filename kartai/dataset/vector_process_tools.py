from kartai.dataset.raster_process_tools import merge_raster_paths
from osgeo import gdal, ogr, osr
import os


def vectorize_raster_gdal_target(gdal_target, output_dir, output_layer_name, crs_number=25832):

    #  create output datasource
    drv = ogr.GetDriverByName("GeoJSON")
    output_path = os.path.join(
        output_dir, output_layer_name + ".geojson")

    dst_ds = drv.CreateDataSource(output_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs_number)  # 25832

    dst_layer = dst_ds.CreateLayer(output_layer_name, srs)

    gdal.Polygonize(gdal_target.GetRasterBand(1), gdal_target.GetRasterBand(1),
                    dst_layer, -1, [], callback=None)

    dst_ds.Destroy()

    return dst_layer

    # mosaic_res = None  # Close file and flush to disk


def vectorize_raster_paths(paths, output_dir, output_layer_name, crs_number=25832):
    mosaic_res = merge_raster_paths(paths)

    #  create output datasource
    drv = ogr.GetDriverByName("GeoJSON")
    output_path = os.path.join(
        output_dir, output_layer_name + ".geojson")

    dst_ds = drv.CreateDataSource(output_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs_number)

    dst_layer = dst_ds.CreateLayer(output_layer_name, srs)

    gdal.Polygonize(mosaic_res.GetRasterBand(1), mosaic_res.GetRasterBand(1),
                    dst_layer, -1, [], callback=None)

    return dst_layer

    # dst_ds.Destroy()

    # mosaic_res = None  # Close file and flush to disk
