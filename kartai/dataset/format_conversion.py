import numpy as np
from PIL import Image
from osgeo import gdal, gdal_array


def image_to_np(image):
    return np.asarray(image)


def np_to_image(np_array):
    return Image.fromarray(np_array)


def gdal_to_np(gdal_ds):
    return np.array(gdal_ds.GetRasterBand(1).ReadAsArray())


def np_to_gdal_mem_dep(np_data, geoTransform, projection):
    # Save in memory instead of to file
    drv = gdal.GetDriverByName("MEM")
    width = np_data.shape[1]
    height = np_data.shape[0]
    target = drv.Create('mem', width, height, 1, gdal.GDT_Byte)
    target.SetGeoTransform(geoTransform)
    target.SetProjection(projection)
    target.GetRasterBand(1).WriteArray(np_data)
    return target


def np_to_gdal_mem(np_data, geo_transform=None, projection=None):
    """Save numpy data to a gdal dataset in memory or to file"""
    driver_name="MEM"
    file_name="mem"
    drv = gdal.GetDriverByName(driver_name)
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(np_data.dtype)
    channels = np_data.shape[2] if len(np_data.shape) == 3 else 1
    if channels > 1:
        raise ValueError("Method only supports one channel - need rewrite in order support several")

    target = drv.Create(
        file_name, np_data.shape[1], np_data.shape[0], 1, gdal_type)
    target.SetGeoTransform(geo_transform)
    target.SetProjection(projection)
    target.GetRasterBand(1).WriteArray(np.squeeze(np_data))

    return target
