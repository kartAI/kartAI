import os
import sys

import numpy as np
from osgeo import ogr
from PIL import Image

from env import get_env_variable
from kartai.datamodels_and_services.Region import Region
from kartai.utils.geometry_utils import parse_region_arg


def get_X_tuple(batch_size, datapaths_for_batch, tuple_input):
    X_tuple = []

    for img_input in tuple_input:
        dimensions = img_input["dimensions"]
        input_name = img_input['name']

        X_batch = np.empty(
            (batch_size, dimensions[0], dimensions[1], dimensions[2]))

        for i in range(len(datapaths_for_batch)):
            data_instance = None
            try:
                data_instance = Image.open(
                    datapaths_for_batch[i][input_name])
            except BaseException as ex:
                print(
                    f"error opening image: {datapaths_for_batch[i][{input_name}]}, {ex}", file=sys.stderr)

            if data_instance:
                data_instance_array = np.array(data_instance)
                if(len(data_instance_array.shape) == 2):
                    data_instance_array = np.expand_dims(
                        data_instance_array, 2)
                X_batch[i, ] = data_instance_array
            else:
                X_batch[i, ] = np.zeros(
                    (dimensions[0], dimensions[1], dimensions[2]))

        X_tuple.append(X_batch)
    return X_tuple


def get_X_stack(batch_size, datapaths_for_batch, input_stack):
    stack_channels = 0
    first_dim = input_stack[0]["dimensions"][0]
    second_dim = input_stack[0]["dimensions"][1]

    for input in input_stack:
        dimensions = input["dimensions"]
        channels = dimensions[2]
        stack_channels += channels

        if(first_dim != dimensions[0] or second_dim != dimensions[1]):
            print(
                '\n---ERROR: You cannot create an input stack of images with different first and second dimension')
            sys.exit()

    X_batch = np.empty(
        (batch_size, first_dim, second_dim, stack_channels))

    for i in range(len(datapaths_for_batch)):
        X_stack = np.empty((first_dim,
                            second_dim, 0))

        for input in input_stack:
            input_name = input['name']
            channels = input['dimensions'][2]
            data_instance = None
            try:
                data_instance = datapaths_for_batch[i][input_name].array
            except BaseException as ex:
                print(
                    f"error opening image: {datapaths_for_batch[i]}, {ex}", file=sys.stderr)

            if data_instance is not None:
                data_instance_array = data_instance.transpose(
                    (1, 2, 0))  # np.array(data_instance)
                if(len(data_instance_array.shape) == 2):
                    data_instance_array = np.expand_dims(
                        data_instance_array, 2)

                X_stack = np.append(
                    X_stack, data_instance_array, axis=2)
            else:
                X_stack = np.append(X_stack, np.empty(
                    (first_dim, second_dim, channels)), axis=2)

        X_batch[i, ] = X_stack

    return X_batch


def get_ground_truth(batch_size, tileset_for_batch, ground_truth):
    dimensions = ground_truth["dimensions"]

    Y_batch = np.empty(
        (batch_size, dimensions[0], dimensions[1], dimensions[2]), dtype=int)

    for i in range(len(tileset_for_batch)):
        y_img = None
        ground_truth_name = ground_truth["name"]
        try:
            y_img = tileset_for_batch[i][ground_truth_name].array
        except BaseException as ex:
            print(
                f"error opening image: {tileset_for_batch[i]}.ground_truth_name, {ex}", file=sys.stderr)

        if y_img is not None:
            Y_batch[i, :, :, 0] = np.array(y_img)
        else:
            Y_batch[i, :, :, 0] = np.zeros(
                (dimensions[0], dimensions[1]), dtype=int)
    return Y_batch


def validate_model_data_input(datagenerator_config, model_name, segmentation_models):
    input_stack = datagenerator_config["model_input_stack"]
    input_tuple = datagenerator_config["model_input_tuple"]

    if(len(input_tuple) > 2):
        print('\n---ERROR: no models support more than two inputs')
        sys.exit()
    elif((len(input_tuple) > 0) & (model_name not in segmentation_models.models_supporting_tupple_input)):
        print(f'\n---ERROR: {model_name} does not support tuple input')
        sys.exit()

    elif((len(input_stack) > 1) & (model_name not in segmentation_models.models_supporting_stacked_input)):
        print(
            f'\n---ERROR: {model_name} expects tuple input, not stacked input')
        sys.exit()


def check_for_existing_dataset(training_dataset_name):

    if('test_data_teacher' in training_dataset_name):
        return

    created_dataset_dir = get_env_variable("created_datasets_directory")

    output_directory = os.path.join(
        created_dataset_dir, training_dataset_name)

    dataset_name_exist_already = os.path.isdir(output_directory)

    if dataset_name_exist_already:
        print(f'A dataset named {training_dataset_name} already exist')

        wantToOverwrite = input(
            "Do you want to continue? This will mean that the previous dataset will be overwritten. Enter y to continue:")

        if not wantToOverwrite == 'y':
            sys.exit()


def get_dataset_region(tile_grid, region_path=None, x_min=None, y_min=None, x_max=None, y_max=None):

    reg = parse_region_arg(region_path)
    if reg is not None:
        reg = Region(reg)

    if (x_min is not None and y_min is not None and x_max is not None and y_max is not None) or \
            (reg is not None and (x_min is not None or y_min is not None or x_max is not None or y_max is not None)):
        # Create a ring from the boundingbox
        ring = ogr.Geometry(ogr.wkbLinearRing)
        x_min = x_min if x_min is not None else reg.minx
        y_min = y_min if y_min is not None else reg.miny
        x_max = x_max if x_max is not None else reg.maxx
        y_max = y_max if y_max is not None else reg.maxy
        ring.AddPoint(x_min, y_min)
        ring.AddPoint(x_max, y_min)
        ring.AddPoint(x_max, y_max)
        ring.AddPoint(x_min, y_max)
        ring.AddPoint(x_min, y_min)

        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometryDirectly(ring)
        poly.AssignSpatialReference(tile_grid.srs)

        if reg is None:
            reg = Region(poly)
        else:
            reg = Region(reg.poly.Intersection(poly))
    return reg
