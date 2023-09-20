import json
from osgeo import ogr, osr
import os


def parse_feature_region(feature, from_CRS=25832, swap_coords=False) -> ogr.Geometry:
    if swap_coords:
        feature = swap_coordinates_in_polygon_feature(feature)

    region = ogr.CreateGeometryFromJson(json.dumps(feature['geometry']))
    out_CRS = 25832
    if from_CRS != out_CRS:
        transform(region, from_CRS, out_CRS)

    return region


def swap_coordinates_in_polygon_feature(feature):
    # Only support polygon for now
    if feature['geometry']['type'] != "Polygon":
        raise ValueError(
            f"Only supports polygon, but {feature['geometry']['type']} was passed")
    coordinates = feature['geometry']['coordinates'][0]
    swapped_coordinates = []
    for coord in coordinates:
        swapped_coordinates.append([coord[1], coord[0]])
    feature['geometry']['coordinates'][0] = swapped_coordinates
    return feature


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
