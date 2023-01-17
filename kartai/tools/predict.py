import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
from azure import blobstorage
from osgeo import gdal, ogr, osr
from PIL import Image
from tensorflow import keras

import env
from kartai.datamodels_and_services.ImageSourceServices import Tile
from kartai.exceptions import CheckpointNotFoundException, InvalidCheckpointException
from kartai.utils.model_utils import get_ground_truth, load_checkpoint_model, _get_input_images, checkpoint_exist
from kartai.tools.train import getOptimizer
from kartai.utils.train_utils import get_existing_model_names

from kartai.metrics.meanIoU import (Iou_point_5, Iou_point_6, Iou_point_7,
                                    Iou_point_8, Iou_point_9, IoU, IoU_fz)


def predict_and_evaluate(created_datasets_path, datagenerator_config, checkpoint_name_to_predict_with, save_prediction_images=True, save_diff_images=True, generate_metadata=True, dataset_to_evaluate="test"):

    has_checkpoint = checkpoint_exist(checkpoint_name_to_predict_with)
    if(not has_checkpoint):
        raise CheckpointNotFoundException("Checkpoint does not exist")
    try:
        model = load_checkpoint_model(checkpoint_name_to_predict_with)
    except:
        raise InvalidCheckpointException(
            'Failed to load checkpoint {checkpoint_to_predict_with}, most likely du to empty files and no saved weights')

    with open(os.path.join(created_datasets_path, f"{dataset_to_evaluate}_set.json"), "r") as file:
        input_list_dataset_json = json.load(file)
        input_paths = Tile.tileset_from_json(input_list_dataset_json)

    input_images = _get_input_images(
        input_paths, datagenerator_config)

    batch_size = len(input_paths)
    input_labels = get_ground_truth(
        batch_size, input_paths, datagenerator_config["ground_truth"])

    opt = getOptimizer("RMSprop", False)

    batch_size = 2 # 6
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryAccuracy(), IoU, IoU_fz, Iou_point_5, Iou_point_6, Iou_point_7, Iou_point_8, Iou_point_9])

    # Evaluate model on test data
    results = model.evaluate(input_images, input_labels, batch_size, return_dict=True)
    # Predict test data from model
    predictions = model.predict(input_images, batch_size)

    output_dir = os.path.join(env.get_env_variable(
        'prediction_results_directory'), checkpoint_name_to_predict_with)

    save_outputs = save_prediction_images or save_diff_images or generate_metadata
    if save_outputs:
        output_test_dir = os.path.join(env.get_env_variable(
            'prediction_results_directory'), checkpoint_name_to_predict_with)
        os.makedirs(output_test_dir, exist_ok=True)

    if save_prediction_images or save_diff_images:
        _save_outputs(output_test_dir, predictions, input_paths,
                      input_labels, save_prediction_images, save_diff_images)

    if generate_metadata:
        createMetadataFile(created_datasets_path, checkpoint_name_to_predict_with,
                           output_dir, results)

    return results


def _save_outputs(output_test_dir, predictions, input_paths, input_labels, save_prediction_images=True, save_diff_images=True):

    if save_prediction_images:
        file_list = savePredictedImages(predictions, input_paths,
                                        output_test_dir, "_val_predict.tif")
        gdal.BuildVRT(os.path.join(output_test_dir,
                      "val_predict.vrt"), file_list, addAlpha=True)

    if save_diff_images:
        file_list = savePredictedImages(predictions - input_labels,
                                        input_paths, output_test_dir, "_val_diff.tif")
        gdal.BuildVRT(os.path.join(output_test_dir,
                      "val_diff.vrt"), file_list, addAlpha=True)


def savePredictedImages(test_pred, test_input_list, output_test_dir, suffix):
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
    # Export test data as geotiff
    file_list = []
    for i in range(len(test_pred)):
        input_img_name = Path(test_input_list[i]['image'].file_path).stem
        # test_sample = gdal.Open(test_input_list[i]['image'])
        predict_fn = input_img_name + suffix
        predict_fn = os.path.join(output_test_dir, predict_fn)
        file_list.append(os.path.abspath(predict_fn))
        ds = gdal.GetDriverByName('GTiff').Create(predict_fn,
                                                  test_pred.shape[1], test_pred.shape[2], 1, gdal.GDT_Float32,
                                                  ['COMPRESS=LZW', 'PREDICTOR=2'])
        tranformation = test_input_list[i]['image'].geo_transform
        ds.SetGeoTransform(tranformation)
        try:
            projection = test_input_list[i]['image'].srs_wkt
        except:
            # Temp error fix due to proj library error
            print("proj error getting projection - fallback to epsg:25832")
            projection = ("EPSG:25832")

        ds.SetProjection(projection)
        ds.GetRasterBand(1).WriteArray(test_pred[i, :, :, 0])
        ds = None
    return file_list


def createMetadataFile(created_datasets_path, checkpoint_path, output_test_dir, results):
    ct = datetime.datetime.now()
    # TODO: fetch training dataset id instead of adding path

    meta = {"test dataset path:": str(created_datasets_path),
            "checkpoint name:": str(checkpoint_path),
            "results": results,
            "prediction date:": str(ct),
            }
    ident = 2

    prediction_file = output_test_dir+'_prediction.json'

    with open(prediction_file, 'w') as outfile:
        json.dump(meta, outfile, indent=ident)

    print(json.dumps(meta,  indent=ident))


def add_parser(subparser):
    parser = subparser.add_parser(
        "predict",
        help="compare images, labels and masks side by side",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    existing_trained_model_names = get_existing_model_names()

    parser.add_argument('-dn', '--dataset_name', type=str,
                        help='Name for training dataset', required=True)
    parser.add_argument('-cn', '--checkpoint_name', type=str, choices=existing_trained_model_names,
                        help='Name for checkpoint file to use for prediction', required=True)
    parser.add_argument('-s', '--save_prediction_images', action="store_true",
                        help='Wether or not to save predictions as images', default=True)
    parser.add_argument('-diff', '--save_difference_images', action="store_true",
                        help='Wether or not to save differences between labels and predictions as images')
    parser.add_argument('-c', '--config', type=str,
                        help='Path to data generator configuration', default='config/ml_input_generator/ortofoto.json')

    parser.set_defaults(func=main)


def main(args):

    created_datasets_dir = os.path.join(env.get_env_variable(
        'created_datasets_directory'), args.dataset_name)
    checkpoint_path = os.path.join(env.get_env_variable(
        'trained_models_directory'), args.checkpoint_name+'.h5')

    if not os.path.isfile(checkpoint_path):
        blobstorage.downloadModelFileFromAzure(args.checkpoint_name)

    with open(args.config, encoding="utf8") as config:
        datagenerator_config = json.load(config)

    predict_and_evaluate(created_datasets_dir, datagenerator_config, args.checkpoint_name,
                         args.save_prediction_images, args.save_difference_images)
