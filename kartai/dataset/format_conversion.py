import numpy as np
from PIL import Image
from osgeo import gdal


def image_to_np(image):
    return np.asarray(image)


def np_to_image(np_array):
    return Image.fromarray(np_array)


def gdal_to_np(gdal_ds):
    return np.array(gdal_ds.GetRasterBand(1).ReadAsArray())


def np_to_gdal_mem(np_data, input_ds):
    # Save in memory instead of to file
    drv = gdal.GetDriverByName("MEM")
    width = np_data.shape[1]
    height = np_data.shape[0]
    target = drv.Create('mem', width, height, 1, gdal.GDT_Byte)
    target.SetGeoTransform(input_ds.GetGeoTransform())
    target.SetProjection(input_ds.GetProjectionRef())
    target.GetRasterBand(1).WriteArray(np_data)
    return target
