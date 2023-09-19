import json
from osgeo import ogr, osr
import os


def parse_feature_region(feature, from_CRS=25832):
    region = ogr.CreateGeometryFromJson(json.dumps(feature['geometry']))
    print("region", region)
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


def main():
    feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "coordinates": [
                        [
                            [
                                10.692701920308451,
                                59.980359635142975
                            ],
                            [
                                10.64428002426655,
                                59.962185657850995
                            ],
                            [
                                10.687321709637587,
                                59.9477066531534
                            ],
                            [
                                10.718257920997559,
                                59.964878728269525
                            ],
                            [
                                10.692701920308451,
                                59.980359635142975
                            ]
                        ]
                    ],
                    "type": "Polygon"
                }
            }
        ]
    }

    reg = parse_feature_region(feature, 4326)
    print(reg)


if __name__ == "__main__":
    main()
