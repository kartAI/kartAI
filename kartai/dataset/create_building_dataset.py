import glob
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image
import geopandas as gp
import pandas as pd
import numpy as np
from azure.blobstorage import uploadBuildingsDetectionDataset


import env
import rasterio
import rasterio.features
import rasterio.merge

from rasterstats import zonal_stats
from kartai.datamodels_and_services.ImageSourceServices import Tile
from kartai.utils.crs_utils import get_defined_crs_from_config, get_defined_crs_from_config_path

from kartai.utils.dataset_utils import get_X_tuple

default_output_dir = os.path.join(env.get_env_variable(
    'prediction_results_directory'), 'prediction_for_client')

output_prediction_suffix = "prediction"
default_output_predictions_name = f"_{output_prediction_suffix}.tif"

# Used by API


def create_predicted_buildings_dataset(geom, checkpoint_name, data_config_path, output_dir=default_output_dir):
    skip_to_postprocess = False  # For testing

    if skip_to_postprocess == False:
        run_ml_predictions(checkpoint_name, output_dir,
                           default_output_predictions_name, data_config_path, geom)

        time.sleep(2)  # Wait for complete saving to disk

    print('Starting postprocess')
    predictions_path = sorted(
        glob.glob(output_dir+f"/*{default_output_predictions_name}"))

    crs = get_defined_crs_from_config_path(data_config_path)
    all_predicted_buildings_dataset = get_all_predicted_buildings_dataset(
        predictions_path, crs)
    return all_predicted_buildings_dataset


def create_building_dataset(geom, checkpoint_name, area_name, data_config_path, only_raw_predictions, skip_to_postprocess, output_dir=default_output_dir, max_mosaic_batch_size=200, save_to='azure'):

    with open(data_config_path, "r") as config_file:
        config = json.load(config_file)

    if not skip_to_postprocess:
        run_ml_predictions(checkpoint_name, output_dir,
                           default_output_predictions_name, data_config_path, geom)

        time.sleep(2)  # Wait for complete saving to disk

    print('Starting postprocess')

    produce_resulting_datasets(
        output_dir, config, max_mosaic_batch_size, only_raw_predictions, f"{area_name}_{checkpoint_name}", save_to)


def save_dataset(data, filename, output_dir, modelname, save_to):
    if(save_to == 'azure'):
        save_dataset_to_azure(data, filename, modelname)
    else:
        save_dataset_locally(data, filename, output_dir)


def save_dataset_locally(data, filename, output_dir):
    file = open(os.path.join(
        output_dir, filename), 'w')
    file.write(data)
    file.close()


def save_dataset_to_azure(data, filename, modelname):
    from azure.storage.blob import BlobServiceClient, __version__
    connect_str = env.get_env_variable('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(
        connect_str)
    uploadBuildingsDetectionDataset(
        modelname, filename, blob_service_client, data)


def run_ml_predictions(checkpoint_name, output_dir, output_predictions_name=default_output_predictions_name, config_path=None, geom=None, skip_data_fetching=False, dataset_path_to_predict=None, tupple_data=False):
    from azure import blobstorage
    from tensorflow import keras
    from kartai.tools.predict import savePredictedImages
    from kartai.tools.train import getLoss
    from kartai.dataset.PredictionArea import fetch_data_to_predict

    from kartai.metrics.meanIoU import (IoU, IoU_fz, Iou_point_5, Iou_point_6,
                                        Iou_point_7, Iou_point_8, Iou_point_9)

    # Create ortofoto tiles for bbox area
    if skip_data_fetching == False:
        fetch_data_to_predict(geom, config_path)

    checkpoint_path = os.path.join(env.get_env_variable(
        'trained_models_directory'), checkpoint_name+".h5")

    if not os.path.isfile(checkpoint_path):
        blobstorage.downloadModelFileFromAzure(
            Path(checkpoint_name).stem)

    dependencies = {
        'BinaryFocalLoss': getLoss('focal_loss'),
        'Iou_point_5': Iou_point_5,
        'Iou_point_6': Iou_point_6,
        'Iou_point_7': Iou_point_7,
        'Iou_point_8': Iou_point_8,
        'Iou_point_9': Iou_point_9,
        'IoU': IoU,
        'IoU_fz': IoU_fz
    }
    model = keras.models.load_model(
        checkpoint_path, custom_objects=dependencies)

    # Read file with references to created ortofoto images that should be analyzed
    # Prediction data is without height info, therefor crashes

    if(not dataset_path_to_predict):
        dataset_path_to_predict = os.path.join(env.get_env_variable(
            'created_datasets_directory'), 'prediction_set.json')

    with open(dataset_path_to_predict) as f:
        prediction_input_list = Tile.tileset_from_json(json.load(f))

    if not tupple_data:
        if("image" in prediction_input_list[0] and "lidar" in prediction_input_list[0]):
            img_dims = [512, 512, 4]
        elif("image" in prediction_input_list[0]):
            img_dims = [512, 512, 3]
        elif("lidar" in prediction_input_list[0]):
            img_dims = [512, 512, 1]
        else:
            sys.exit("Unknown input type dimensions")

    batch_size = min(8, len(prediction_input_list)-1)
    num_predictions = len(prediction_input_list)
    splits = (num_predictions//batch_size) if num_predictions % batch_size == 0 else (
        num_predictions//batch_size) + 1

    for i in range(splits):
        print(
            f'Run batch {i} of {splits}. Instances {batch_size*i} to {batch_size*i+batch_size}.')
        input_batch = prediction_input_list[batch_size *
                                            i:batch_size*i+batch_size]
        print('batch', input_batch)
        # Generates stack of images as an array with shape (batch_size x height x length x channels)
        if tupple_data:
            input_tuples = []
            tuple1 = {
                "name": "image",
                "dimensions": [512, 512, 3]
            }
            tuple2 = {
                "name": "lidar",
                "dimensions": [512, 512, 1]
            }
            input_tuples.append(tuple1)
            input_tuples.append(tuple2)

            tupples_to_predict = get_X_tuple(
                len(input_batch), input_batch, input_tuples)

        else:
            images_to_predict = np.empty(
                (len(input_batch), img_dims[0], img_dims[1], img_dims[2]))
            for i_batch in range(len(input_batch)):
                # Open image
                print('open image', input_batch[i_batch])
                gdal_image = input_batch[i_batch]['image'].array
                image = gdal_image.transpose((1, 2, 0))
                if('lidar' in input_batch[i_batch]):
                    lidar = input_batch[i_batch]['lidar'].array.reshape(
                        512, 512, 1)
                    combined_arr = np.concatenate((image, lidar), axis=2)
                    images_to_predict[i_batch, ] = combined_arr
                else:
                    images_to_predict[i_batch, ] = image

        # If lidar images => add the lidar channel to the image to predict as an extra channel
        if tupple_data:
            np_pred_results_iteration = model.predict(tupples_to_predict)
        else:
            np_pred_results_iteration = model.predict(images_to_predict)

        savePredictedImages(np_pred_results_iteration, input_batch,
                            output_dir, output_predictions_name)
    print('Completed predictions, start postprocessing')


def produce_resulting_datasets(output_dir, config, max_batch_size, only_raw_predictions, modelname, save_to):

    predictions_path = sorted(
        glob.glob(output_dir+f"/*{default_output_predictions_name}"))
    print('output_dir', output_dir)
    print('num predictions', len(predictions_path))
    print('output_predictions_name', default_output_predictions_name)

    batch_size = min(max_batch_size, len(predictions_path))
    num_splits = (len(predictions_path) // batch_size)

    if num_splits*batch_size < len(predictions_path):
        num_splits += 1  # Need to add an extra run to process the remaining images

    for i in range(num_splits):
        print(
            f'Starting post processing of resulting labels, iteration {i} of {num_splits}, images {i*batch_size} to {i*batch_size+batch_size}')  # image 67400 to 67600
        if(i == num_splits):
            # Last run, batch size might be lower
            batch_prediction_paths = predictions_path[i *
                                                      batch_size:(len(predictions_path)-1)]
        else:
            batch_prediction_paths = predictions_path[i *
                                                      batch_size:i*batch_size+batch_size]

        if only_raw_predictions == False:
            print('first in batch predictions', batch_prediction_paths[0])
            # Check if there are data in this batch
            new_tilbygg_dataset, new_frittliggende_bygg_dataset, existing_buildings_dataset, all_predicted_buildings_dataset = create_datasets(
                batch_prediction_paths, output_dir, only_raw_predictions, config)

            if all_predicted_buildings_dataset:  # Check if there are data in this batch
                save_dataset(
                    all_predicted_buildings_dataset, f'raw_predictions_{str(i)}.json', output_dir, modelname, save_to)

                save_dataset(new_tilbygg_dataset,
                             f'new_tilbygg_{str(i)}.json', output_dir, modelname, save_to)
                save_dataset(new_frittliggende_bygg_dataset,
                             f'new_frittliggende_bygg_{str(i)}.json', output_dir, modelname, save_to)
                save_dataset(
                    existing_buildings_dataset, f'existing_bygg_{str(i)}.json', output_dir, modelname, save_to)
                # Free memory
                del all_predicted_buildings_dataset
                del new_tilbygg_dataset
                del new_frittliggende_bygg_dataset
                del existing_buildings_dataset
            else:
                print('no data in batch')
        else:
            all_predicted_buildings_dataset = create_datasets(
                batch_prediction_paths, output_dir, only_raw_predictions, config)
            if all_predicted_buildings_dataset:
                save_dataset(all_predicted_buildings_dataset,
                             f'raw_predictions_{str(i)}.json', output_dir, modelname, save_to)
                # Free memory
                del all_predicted_buildings_dataset

    print('---PROCESS COMPLETED')


def create_normalized_prediction_mask(img):
    arr = img.read(1)
    mask = (arr > 0.5).astype(dtype=np.int32)
    return mask


def polygonize_mask(mask, img, crs):
    shapes = rasterio.features.shapes(
        mask, connectivity=4, transform=img.transform)

    records = [{"geometry": geometry, "properties": {"value": value}}
               for (geometry, value) in shapes if value == 1]
    geoms = list(records)
    polygons = gp.GeoDataFrame.from_features(geoms, crs=crs)
    return polygons


def get_raw_predictions(predictions_path):
    raw_prediction_imgs = []

    for i in range(len(predictions_path)):
        raw_prediction_img = rasterio.open(predictions_path[i])
        raw_prediction_imgs.append(raw_prediction_img)

    return raw_prediction_imgs


def create_datasets(batch_predictions_path, output_dir, only_raw_predictions, config):

    crs = get_defined_crs_from_config(config)

    predictions_path = batch_predictions_path

    all_predicted_buildings_dataset = get_all_predicted_buildings_dataset(
        predictions_path, crs)

    labels_dataset = get_labels_dataset(
        predictions_path, output_dir, config, crs)

    if labels_dataset.empty or all_predicted_buildings_dataset.empty:
        if only_raw_predictions == False:
            return None, None, None, None
        else:
            return None

    new_buildings_dataset = get_new_buildings_dataset(
        all_predicted_buildings_dataset, labels_dataset)

    # Setting labels area in order to filter the following datasets
    all_predicted_buildings_dataset['labels_area'] = all_predicted_buildings_dataset.geometry.area - \
        new_buildings_dataset.geometry.area

    if only_raw_predictions == False:
        new_frittliggende_bygg_dataset = get_new_frittliggende_dataset(
            all_predicted_buildings_dataset)

        new_tilbygg_dataset = get_tilbygg_dataset(
            all_predicted_buildings_dataset, labels_dataset)

        existing_buildings = get_existing_buildings_dataset(
            all_predicted_buildings_dataset,  new_tilbygg_dataset, new_frittliggende_bygg_dataset)

    # justering på datasettene

    raw_prediction_imgs = get_raw_predictions(predictions_path)
    full_img, full_transform = rasterio.merge.merge(raw_prediction_imgs)

    if only_raw_predictions == False:
        new_tilbygg_dataset = perform_last_adjustments(
            new_tilbygg_dataset, full_img, full_transform, crs)
        new_frittliggende_bygg_dataset = perform_last_adjustments(
            new_frittliggende_bygg_dataset, full_img, full_transform, crs)
        existing_buildings = perform_last_adjustments(
            existing_buildings, full_img, full_transform, crs)

    all_predicted_buildings_dataset = add_probability_values(
        all_predicted_buildings_dataset, full_img, full_transform, crs)

    if only_raw_predictions == True:
        return all_predicted_buildings_dataset.to_json()
    else:
        return new_tilbygg_dataset.to_json(), new_frittliggende_bygg_dataset.to_json(), existing_buildings.to_json(), all_predicted_buildings_dataset.to_json()


def perform_last_adjustments(dataset, full_img, full_transform, crs):
    if dataset.empty:
        return None
    simplified_dataset = simplify_dataset(dataset)
    annotated_dataset = add_probability_values(
        simplified_dataset, full_img, full_transform, crs)
    assign_crs(annotated_dataset, crs)
    return annotated_dataset


def assign_crs(dataframe, crs):
    dataframe.set_crs(crs)


def add_probability_values(dataset, full_img, full_transform, crs):
    dataset = dataset.reset_index(drop=True)
    probabilites = zonal_stats(dataset.geometry,
                               full_img[0], affine=full_transform, stats='mean')

    dataset['prob'] = gp.GeoDataFrame(probabilites, crs=crs)
    return dataset


def simplify_dataset(dataset):
    # Remove all small buildings in dataset
    dataset = dataset.loc[dataset.geometry.area > 1]

    # Remove points in polygons to get better building-look
    dataset['geometry'] = dataset.simplify(
        tolerance=0.25)

    return dataset


def get_existing_buildings_dataset(all_predicted_buildings_dataset, new_tilbygg_dataset, new_frittliggende_bygg_dataset):
    demolished_buildings = gp.overlay(
        all_predicted_buildings_dataset, new_tilbygg_dataset, how='difference')
    demolished_buildings = gp.overlay(
        demolished_buildings, new_frittliggende_bygg_dataset, how='difference')

    return demolished_buildings


def get_tilbygg_dataset(all_predicted_buildings_dataset, labels_dataset):
    predictions_minus_labels = gp.overlay(
        all_predicted_buildings_dataset, labels_dataset, how='difference')

    new_tilbygg_dataset = predictions_minus_labels.loc[
        predictions_minus_labels['labels_area'] > 0]

    # Clean up
    minimized_new_tilbygg_dataset = new_tilbygg_dataset.geometry.buffer(-1)
    boundary_dataset = minimized_new_tilbygg_dataset.loc[minimized_new_tilbygg_dataset.area > 0].buffer(
        1.5)

    new_tilbygg_dataset = new_tilbygg_dataset.loc[boundary_dataset.index]
    new_tilbygg_dataset = gp.clip(new_tilbygg_dataset, boundary_dataset)

    new_tilbygg_dataset['Type'] = 2

    return new_tilbygg_dataset


def get_new_frittliggende_dataset(all_predicted_buildings_dataset):
    new_frittliggende_bygg_dataset = all_predicted_buildings_dataset.loc[
        all_predicted_buildings_dataset['labels_area'] == 0]
    new_frittliggende_bygg_dataset['Type'] = 1
    return new_frittliggende_bygg_dataset


def get_new_buildings_dataset(all_predicted_buildings_dataset, labels_dataset):
    new_buildings = gp.overlay(
        all_predicted_buildings_dataset, labels_dataset, how='difference')
    new_buildings.index = new_buildings['b_id']
    return new_buildings


def get_labels_dataset(predictions_path, output_dir, config, crs):
    label_tiles = gp.GeoDataFrame()

    for prediction_path in predictions_path:
        label_path = get_label_for_prediction_path(
            prediction_path,  output_dir, config)
        label = rasterio.open(label_path)
        label_mask = label.read(1)
        label_polygons = polygonize_mask(label_mask, label, crs)
        label_tiles = label_tiles.append(label_polygons, ignore_index=True)

    if label_tiles.empty:
        return label_tiles
    label_tiles['geometry'] = get_valid_geoms(label_tiles)
    all_labels_dissolved = merge_connected_geoms(label_tiles)
    return all_labels_dissolved


def get_all_predicted_buildings_dataset(predictions_path, crs):

    prediction_tiles = gp.GeoDataFrame()

    for prediction_path in predictions_path:
        raw_prediction_img = rasterio.open(
            prediction_path)

        prediction_mask = create_normalized_prediction_mask(raw_prediction_img)
        prediction_polygons = polygonize_mask(
            prediction_mask, raw_prediction_img, crs)

        prediction_tiles = pd.concat(
            [prediction_tiles, prediction_polygons], ignore_index=True)

    if prediction_tiles.empty:
        return prediction_tiles

    prediction_tiles['geometry'] = get_valid_geoms(prediction_tiles)

    dissolved_dataset = merge_connected_geoms(prediction_tiles)
    dissolved_dataset['b_id'] = dissolved_dataset.index
    dissolved_dataset['area'] = dissolved_dataset.geometry.area
    dissolved_dataset['Type'] = 0  # Setting class 0 -> existing building

    # Remove all buildings with area less than 1 square meter
    cleaned_dataset = dissolved_dataset.loc[dissolved_dataset.geometry.area > 1]
    assign_crs(cleaned_dataset, crs)
    return cleaned_dataset


def merge_connected_geoms(geoms):
    dissolved_geoms = geoms.dissolve(by='value')
    dissolved_geoms = dissolved_geoms.explode().reset_index(drop=True)
    return dissolved_geoms


def get_valid_geoms(geoms):
    # Hack to clean up geoms
    if geoms.empty:
        return geoms
    return geoms.geometry.buffer(0)


def get_label_for_prediction_path(prediction_path, output_dir, config):
    labels_folder = get_labels_folder(config)
    label_path = prediction_path.replace('_'+output_prediction_suffix, "").replace(
        output_dir, labels_folder)
    return label_path


def get_labels_folder(config):
    tilegrid = config["TileGrid"]
    cache_folder_name = f"{tilegrid['srid']}_{tilegrid['x0']}_{tilegrid['y0']}_{tilegrid['dx']}_{tilegrid['dy']}"
    labelSourceName = get_label_source_name(config)

    byggDb_path = os.path.join(env.get_env_variable(
        "cached_data_directory"), labelSourceName, cache_folder_name, "512")
    return byggDb_path


def get_label_source_name(config):
    labelSourceName = None
    for source in config["ImageSources"]:
        if(source["type"] == "PostgresImageSource"):
            labelSourceName = source["name"]

    if(not labelSourceName):
        raise Exception("Could not find PostgresImageSource in config")
    return labelSourceName
