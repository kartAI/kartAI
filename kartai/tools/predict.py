import argparse
import datetime
import json
import os
from pathlib import Path
from azure import blobstorage
from osgeo import gdal, ogr, osr
from tensorflow import keras
import env
from kartai.datamodels_and_services.ImageSourceServices import Tile
from kartai.utils.confidence import Confidence
from kartai.exceptions import CheckpointNotFoundException, InvalidCheckpointException
from kartai.utils.model_utils import get_ground_truth, load_checkpoint_model, _get_input_images, checkpoint_exist
from kartai.tools.train import get_optimizer
from kartai.utils.train_utils import get_existing_model_names

from kartai.metrics.meanIoU import (Iou_point_5, Iou_point_6, Iou_point_7,
                                    Iou_point_8, Iou_point_9, IoU, IoU_fz)


def predict_and_evaluate(created_datasets_path: str, datagenerator_config: str, checkpoint_name_to_predict_with: str, save_prediction_images: bool = True, save_diff_images: bool = True, generate_metadata: bool = True, dataset_to_evaluate: str = "test", batch_size=6):

    has_checkpoint = checkpoint_exist(checkpoint_name_to_predict_with)
    if (not has_checkpoint):
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

    num_inputs = len(input_paths)
    input_labels = get_ground_truth(
        num_inputs, input_paths, datagenerator_config["ground_truth"])

    opt = get_optimizer("RMSprop", False)

    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryAccuracy(), IoU, IoU_fz, Iou_point_5, Iou_point_6, Iou_point_7, Iou_point_8, Iou_point_9, Confidence()])

    # Evaluate model on test data
    results = model.evaluate(input_images, input_labels,
                             batch_size, return_dict=True)
    # Predict test data from model
    predictions = model.predict(input_images, batch_size)

    output_dir = os.path.join(env.get_env_variable(
        'prediction_results_directory'), checkpoint_name_to_predict_with)

    save_outputs = save_prediction_images or save_diff_images or generate_metadata
    if save_outputs:
        output_dir = os.path.join(env.get_env_variable(
            'prediction_results_directory'), checkpoint_name_to_predict_with)
        os.makedirs(output_dir, exist_ok=True)

    if save_prediction_images or save_diff_images:
        _save_outputs(output_dir, predictions, input_paths,
                      input_labels, save_prediction_images, save_diff_images)

    if generate_metadata:
        create_metadata_file(created_datasets_path, checkpoint_name_to_predict_with,
                             output_dir, results)

    return results


def _save_outputs(output_dir: str, predictions, input_paths: list[dict], input_labels, projection, save_prediction_images=True, save_diff_images=True):

    if save_prediction_images:
        file_list, _ = save_predicted_images_as_geotiff(predictions, input_paths,
                                                        output_dir, "_val_predict.tif", projection)

        gdal.BuildVRT(os.path.join(output_dir,
                      "val_predict.vrt"), file_list, addAlpha=True)

    if save_diff_images:
        file_list, _ = save_predicted_images_as_geotiff(predictions - input_labels,
                                                        input_paths, output_dir, "_val_diff.tif", projection)

        gdal.BuildVRT(os.path.join(output_dir,
                      "val_diff.vrt"), file_list, addAlpha=True)


def save_predicted_images_as_geotiff(np_predictions, data_samples: list[dict], output_dir: str, projection: str, suffix: str = None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export test data as geotiff
    file_list = []
    for i in range(len(np_predictions)):
        input_img_name = Path(data_samples[i]['image'].file_path).stem

        prediction_output_dir = input_img_name + \
            ".tif" if suffix else input_img_name + "_prediction.tif"
        prediction_output_dir = os.path.join(output_dir, prediction_output_dir)
        file_list.append(os.path.abspath(prediction_output_dir))
        ds = gdal.GetDriverByName('GTiff').Create(prediction_output_dir,
                                                  np_predictions.shape[1], np_predictions.shape[2], 1, gdal.GDT_Float32,
                                                  ['COMPRESS=LZW', 'PREDICTOR=2'])

        transformation = get_transformation(
            data_samples[i])
        ds.SetGeoTransform(transformation)
        ds.SetProjection(projection)
        ds.GetRasterBand(1).WriteArray(np_predictions[i, :, :, 0])
        ds = None

    return file_list, projection


def create_contour_result(raster_path: str, output_dir: str, projection: str, fixed_level_count: list[float]):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a virtual raster:
    raster_filenames = os.listdir(raster_path)
    rasters = []
    for filename in raster_filenames:
        rasters.append(os.path.join(raster_path, filename))

    vrt_output_dir = os.path.join(raster_path, "rasters_virtual.vrt")
    vrt_opt = gdal.BuildVRTOptions(addAlpha=True)
    vrt_res = gdal.BuildVRT(vrt_output_dir, rasters, options=vrt_opt)

    prediction_output_dir_geojson = os.path.join(
        output_dir, "complete_contour.json")

    out_geojson_driver = ogr.GetDriverByName("GeoJSON")
    if os.path.exists(prediction_output_dir_geojson):
        os.remove(prediction_output_dir_geojson)

    out_geojson_source = out_geojson_driver.CreateDataSource(
        prediction_output_dir_geojson)

    out_geojson_layer = out_geojson_source.CreateLayer(
        'geojson_contour', osr.SpatialReference(projection))

    # define fields of id and elev
    fieldDef = ogr.FieldDefn("ID", ogr.OFTInteger)
    out_geojson_layer.CreateField(fieldDef)
    fieldDef = ogr.FieldDefn("elev", ogr.OFTReal)
    out_geojson_layer.CreateField(fieldDef)

    gdal.ContourGenerate(srcBand=vrt_res.GetRasterBand(
        1), contourInterval=0.0, contourBase=1.0, fixedLevelCount=fixed_level_count, useNoData=0, noDataValue=0, dstLayer=out_geojson_layer, idField=0, elevField=1)

    out_geojson_source = None


def get_transformation(data_sample: dict):
    transformation = data_sample['image'].geo_transform
    return transformation


def create_metadata_file(created_datasets_path, checkpoint_path, output_dir, results):
    ct = datetime.datetime.now()
    # TODO: fetch training dataset id instead of adding path

    results["sample_confidence"] = results["sample_confidence"].tolist()

    meta = {"test dataset path:": str(created_datasets_path),
            "checkpoint name:": str(checkpoint_path),
            "results": results,
            "prediction date:": str(ct),
            }
    ident = 2

    prediction_file = output_dir+'_prediction.json'

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
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Size of minibatch', default=8)
    parser.set_defaults(func=main)


def main(args):

    created_datasets_dir = os.path.join(env.get_env_variable(
        'created_datasets_directory'), args.dataset_name)
    checkpoint_path = os.path.join(env.get_env_variable(
        'trained_models_directory'), args.checkpoint_name+'.h5')

    if not os.path.isfile(checkpoint_path):
        blobstorage.download_model_file_from_azure(args.checkpoint_name)

    with open(args.config, encoding="utf8") as config:
        datagenerator_config: dict = json.load(config)

    predict_and_evaluate(created_datasets_dir, datagenerator_config, args.checkpoint_name,
                         args.save_prediction_images, args.save_difference_images, args.batch_size)
