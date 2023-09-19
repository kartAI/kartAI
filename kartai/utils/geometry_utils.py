import json
from osgeo import ogr, osr
import os


def parse_feature_region(feature, from_CRS=25832):
    region = ogr.CreateGeometryFromJson(json.dumps(feature['geometry']))
    out_CRS = 25832
    if from_CRS != out_CRS:
        transform(region, from_CRS, out_CRS)

    return region


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


def transform(geom, inSpatialNum, outSpatialNum):
    """ Transform the coordinate system of geom in place. """
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inSpatialNum)

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outSpatialNum)

    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    geom.Transform(coordTrans)
