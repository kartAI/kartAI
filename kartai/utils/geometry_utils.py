from kartai.datamodels_and_services.Region import Region
from osgeo import ogr
import os


def parse_region_arg(region):
    reg = None
    if region is not None:
        geom_txt = region
        if os.path.isfile(region):
            with open(region, "r") as rf:
                geom_txt = rf.read()

        try:
            reg = ogr.CreateGeometryFromJson(geom_txt)
        except:
            pass

        if reg is None:
            reg = ogr.CreateGeometryFromWkt(geom_txt)

    return reg
