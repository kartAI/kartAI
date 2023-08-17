from osgeo import ogr
import os


def parse_region_arg(region: str, format="OGRLayer") -> ogr.Geometry:
    reg = None
    if region is not None:
        geom_txt = region
        if os.path.isfile(region):
            with open(region, "r") as rf:
                geom_txt = rf.read()

                if format == "text":
                    return geom_txt

        try:
            reg = ogr.CreateGeometryFromJson(geom_txt)
        except:
            pass

        if reg is None:
            reg = ogr.CreateGeometryFromWkt(geom_txt)

    return reg
