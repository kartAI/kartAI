import argparse
import collections
import datetime
import json
import math
import os
import random
import sys
import uuid

import numpy as np
from osgeo import gdal, ogr, osr

from env import get_env_variable
from kartai.datamodels_and_services.Region import Region
from kartai.utils.dataset_utils import (check_for_existing_dataset,
                                        get_dataset_region)
from kartai.datamodels_and_services.DatasetBuilder import DatasetBuilder
from kartai.datamodels_and_services.ImageSet import getImageSets
from kartai.datamodels_and_services.ImageSourceServices import ImageSourceFactory


class TileGrid:
    """Specification of the geometry of a tiled image grid"""

    def __init__(self, srid, x0, y0, dx, dy=None):
        """Create an image tile grid in a coordinate system given by srid,
        using x0, y0 as origin and each image tile having sidelength dx, dy.
        Select an origin south-west of your data area so that all tile indices will be positive"""
        self.srid = srid
        self.srs = osr.SpatialReference()
        self.srs.ImportFromEPSG(srid)
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy if dy is not None else dx

    def image_geom(self, i, j):
        """Return the extent if an image tile with grid index i, j as
        minx, miny, maxx, maxy"""
        minx = self.x0 + i * self.dx
        miny = self.y0 + j * self.dy
        maxx = minx + self.dx
        maxy = miny + self.dy
        return minx, miny, maxx, maxy

    def generate_ij(self, region):
        """Generator function for iterating through all i, j indexes for image
        tiles intersecting the specified area.
        Area is currently a rectangle specified by a list or tuple
        [minx, miny, maxx, maxy], but more complex shapes may be allowed in the future"""
        mini = int(math.floor((region.minx - self.x0) / self.dx))
        minj = int(math.floor((region.miny - self.y0) / self.dy))
        maxi = int(math.ceil((region.maxx - self.x0) / self.dx))
        maxj = int(math.ceil((region.maxy - self.y0) / self.dy))

        # Create ring, fill with zero
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 0)

        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometryDirectly(ring)
        poly.AssignSpatialReference(region.poly.GetSpatialReference())

        j = minj
        while j <= maxj:
            i = mini
            while i <= maxi:
                minx, miny, maxx, maxy = self.image_geom(i, j)
                ring.SetPoint_2D(0, minx, miny)
                ring.SetPoint_2D(1, maxx, miny)
                ring.SetPoint_2D(2, maxx, maxy)
                ring.SetPoint_2D(3, minx, maxy)
                ring.SetPoint_2D(4, minx, miny)

                if region.poly.Intersects(poly):
                    yield i, j
                i += 1
            j += 1


def create_metadata_file(id, training_dataset_name, tile_size, area_extent, region, max_size, division, out_folder):
    ct = datetime.datetime.now()

    meta = {"id": str(id),
            "training_dataset_name": str(training_dataset_name),
            "date_time": str(ct),
            "tile_size": str(tile_size),
            "area_extent": area_extent if area_extent[0] != False else False,
            "region": region,
            "max_size": max_size,
            "division training/test/validation": division
            }
    ident = 2
    file = os.path.join(out_folder, training_dataset_name, 'meta.json')

    with open(file, 'w') as outfile:
        json.dump(meta, outfile, indent=ident)

    print("created metadata-file:\n", json.dumps(meta,  indent=ident))


def getTileGrid(config):
    return TileGrid(config["TileGrid"]["srid"],
                    config["TileGrid"]["x0"], config["TileGrid"]["y0"],
                    config["TileGrid"]["dx"], config["TileGrid"]["dy"])


def convert_tileset(tileset):
    conv_tileset = []
    for item in tileset:
        conv_item = {}
        for k, v in item.items():
            conv_item[k] = v.file_path
        conv_tileset.append(conv_item)
    return conv_tileset


def create_training_data(training_dataset_name, config_file_path, confidence_threshold=None, eval_model_checkpoint=None, region=None, x_min=None, x_max=None, y_min=None, y_max=None):
    training_dataset_id = uuid.uuid4()

    cached_data_dir = get_env_variable("cached_data_directory")
    created_dataset_dir = get_env_variable("created_datasets_directory")

    output_directory = os.path.join(
        created_dataset_dir, training_dataset_name)

    check_for_existing_dataset(training_dataset_name)

    with open(config_file_path, encoding="utf8") as f:
        config = json.load(f)

    tile_grid = getTileGrid(config)

    if(region and not os.path.isfile(region)):
        raise Exception(
            "The given region does not exist at given path", region)

    reg = get_dataset_region(tile_grid, region_path=region, x_min=x_min,
                             y_min=y_min, x_max=x_max, y_max=y_max)

    image_sources, train_set, valid_set, test_set = getImageSources(
        config, cached_data_dir, tile_grid, confidence_threshold=confidence_threshold, eval_model_checkpoint=eval_model_checkpoint, region=reg)

    print(
        f"-------creating dataset ------- \n name: {training_dataset_name}")

    # Find general datasets, to be splitted into train / valid / test
    dataset = createDataset(image_sources, config, config["ImageSources"], region=reg,
                            confidence_threshold=confidence_threshold, eval_model_checkpoint=eval_model_checkpoint)
    # Split into training, test, validation
    training_fraction, validation_fraction = 0.8, 0.1

    if "ProjectArguments" in config:
        if "training_fraction" in config["ProjectArguments"]:
            training_fraction = float(
                config["ProjectArguments"]["training_fraction"])
        if "validation_fraction" in config["ProjectArguments"]:
            validation_fraction = float(
                config["ProjectArguments"]["validation_fraction"])

    seed = 1  # by passing same number to Random() the function will give the same output for same input each time
    random.Random(seed).shuffle(dataset)
    split_1 = min(len(dataset), int(len(dataset) * training_fraction))
    split_2 = min(len(dataset), int(
        len(dataset) * (training_fraction + validation_fraction)))

    # Add to presplitted train / valid / test
    if split_1 > 0:
        train_set += dataset[:split_1]
    if split_1 < split_2:
        valid_set += dataset[split_1:split_2]
    if split_2 < len(dataset):
        test_set += dataset[split_2:]

    # Export file
    data_path = output_directory
    os.makedirs(data_path, exist_ok=True)
    vrt_opt = gdal.BuildVRTOptions(addAlpha=True)
    if train_set:
        train_set = convert_tileset(train_set)
        with open(os.path.join(data_path, "train_set.json"), "w", encoding="utf8") as file:
            json.dump(train_set, file)
        image_channels = collections.defaultdict(list)
        for itm in train_set:
            for k, v in itm.items():
                image_channels[k].append(os.path.abspath(v))
        for k, v in image_channels.items():
            gdal.BuildVRT(os.path.join(
                data_path, f"train_{k}.vrt"), v, options=vrt_opt)
    if valid_set:
        valid_set = convert_tileset(valid_set)
        with open(os.path.join(data_path, "valid_set.json"), "w", encoding="utf8") as file:
            json.dump(valid_set, file)
        image_channels = collections.defaultdict(list)
        for itm in valid_set:
            for k, v in itm.items():
                image_channels[k].append(os.path.abspath(v))
        for k, v in image_channels.items():
            gdal.BuildVRT(os.path.join(
                data_path, f"valid_{k}.vrt"), v, options=vrt_opt)
    if test_set:
        test_set = convert_tileset(test_set)
        with open(os.path.join(data_path, "test_set.json"), "w", encoding="utf8") as file:
            json.dump(test_set, file)
        image_channels = collections.defaultdict(list)
        for itm in test_set:
            for k, v in itm.items():
                image_channels[k].append(os.path.abspath(v))
        for k, v in image_channels.items():
            gdal.BuildVRT(os.path.join(
                data_path, f"test_{k}.vrt"), v, options=vrt_opt)

    # Find first tile_size
    tile_size = 0
    for img_set in (config["ImageSets"] if "ImageSets" in config else []) + \
                   (config["TrainingSet"]["ImageSets"] if "TrainingSet" in config and "ImageSets" in config[
                       "TrainingSet"] else []) + \
                   (config["ValidationSet"]["ImageSets"] if "ValidationSet" in config and "ImageSets" in config[
                       "ValidationSet"] else []) + \
                   (config["TestSet"]["ImageSets"] if "TestSet" in config and "ImageSets" in config[
                       "TestSet"] else []):
        if "tile_size" in img_set:
            tile_size = img_set["tile_size"]
        if tile_size > 0:
            break

    max_size = "None"
    if("ProjectArguments" in config and "max_size" in config["ProjectArguments"]):
        max_size = config["ProjectArguments"]["max_size"]

  #id, training_dataset_name, tile_size, area_extent, region, max_size, division, out_folder
    create_metadata_file(training_dataset_id,
                         training_dataset_name, tile_size,
                         [x_min, y_min, x_max, y_max],
                         region,
                         max_size,
                         f"{training_fraction:.1}/{(1-training_fraction-validation_fraction):.1}/{validation_fraction:.1}",
                         created_dataset_dir)

    return output_directory


def createDataset(image_sources, config, image_source_config, region, confidence_threshold=None, eval_model_checkpoint=None):
    if "ImageSets" not in config or not config["ImageSets"]:
        print("No image sets")
        return []

    if "ProjectArea" in config:
        poly = None
        polytext = config["ProjectArea"]
        if not isinstance(polytext, str):
            polytext = json.dumps(polytext)

        poly = ogr.CreateGeometryFromJson(polytext)
        if poly is None:
            poly = ogr.CreateGeometryFromWkt(polytext)

        if poly is None and "x_min" in config["ProjectArea"] and "y_min" in config["ProjectArea"] and \
                "x_max" in config["ProjectArea"] and "y_max" in config["ProjectArea"]:
            minx = config["ProjectArea"]["x_min"]
            miny = config["ProjectArea"]["y_min"]
            maxx = config["ProjectArea"]["x_max"]
            maxy = config["ProjectArea"]["y_max"]

            # Create ring
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(minx, miny)
            ring.AddPoint(maxx, miny)
            ring.AddPoint(maxx, maxy)
            ring.AddPoint(minx, maxy)
            ring.AddPoint(minx, miny)

            # Create polygon
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometryDirectly(ring)
            poly.AssignSpatialReference(region.poly.GetSpatialReference())

            # Create new region as intersection
            if region is None:
                region = Region(poly)
            else:
                region = Region(region.poly.Intersection(poly))

    if region is None:
        print('No region defined for image set, neither in command line arguments nor in "ProjectArea"', file=sys.stderr)
        exit(-1)

    image_sets = getImageSets(config, image_sources)
    dataset_builder = DatasetBuilder(image_sets)
    if("ProjectArguments" in config):
        return list(dataset_builder.assemble_data(region, image_source_config, project_config=config["ProjectArguments"], confidence_threshold=confidence_threshold, eval_model_checkpoint=eval_model_checkpoint))
    else:
        return list(dataset_builder.assemble_data(region, image_source_config, confidence_threshold=confidence_threshold, eval_model_checkpoint=eval_model_checkpoint))


def getImageSources(config, cache_root, tile_grid, confidence_threshold=None, eval_model_checkpoint=None, region=None):
    image_sources = {}
    for source_config in config["ImageSources"]:
        source = ImageSourceFactory.create(
            cache_root, tile_grid, image_sources, source_config)
        if source:
            image_sources[source_config["name"]] = source
        else:
            print("Unknown source:", source_config, file=sys.stderr)

    # Find presplitted train / valid / test
    train_set, valid_set, test_set = [], [], []
    if "TrainingSet" in config:
        train_set = createDataset(
            image_sources, config["TrainingSet"], config["ImageSources"], region=region, confidence_threshold=confidence_threshold, eval_model_checkpoint=eval_model_checkpoint)
    if "ValidationSet" in config:
        valid_set = createDataset(
            image_sources, config["ValidationSet"], config["ImageSources"], region=region, confidence_threshold=confidence_threshold, eval_model_checkpoint=eval_model_checkpoint)
    if "TestSet" in config:
        test_set = createDataset(
            image_sources, config["TestSet"], config["ImageSources"], region=region, confidence_threshold=confidence_threshold, eval_model_checkpoint=eval_model_checkpoint)

    return image_sources, train_set, valid_set, test_set


def add_parser(subparser):
    parser = subparser.add_parser(
        "create_training_data",
        help="compare images, labels and masks side by side",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-n', '--training_dataset_name', type=str,
                        help='Name for trainingdataset', required=True)
    parser.add_argument("-c", "--config_file", type=str,
                        help="Data configuration file", required=True)
    parser.add_argument("--x_min", type=float,
                        help="Western boundary of training data", required=False)
    parser.add_argument("--y_min", type=float,
                        help="Southern boundary of training data", required=False)
    parser.add_argument("--x_max", type=float,
                        help="Eastern boundary of training data", required=False)
    parser.add_argument("--y_max", type=float,
                        help="Northern boundary of training data", required=False)
    parser.add_argument("--region", type=str,
                        help="Polygon boundary of training area\n"
                             "WKT, json text or filename\n"
                             "alternative to bounding box",
                        required=False)

    parser.set_defaults(func=main)


def main(args):
    create_training_data(args.training_dataset_name, args.config_file, region=args.region,
                         x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max)
