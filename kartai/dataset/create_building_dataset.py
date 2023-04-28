import glob
import json
import os
import sys
import time
from pathlib import Path
import geopandas as gp
import pandas as pd
import numpy as np
from azure.blobstorage import upload_data_to_azure


import env
import rasterio
import rasterio.features
import rasterio.merge

from rasterstats import zonal_stats
from kartai.datamodels_and_services.ImageSourceServices import Tile
from kartai.utils.crs_utils import get_defined_crs_from_config, get_defined_crs_from_config_path, get_projection_from_config_path

from kartai.utils.dataset_utils import get_X_tuple
from kartai.utils.prediction_utils import get_raster_predictions_dir, get_vector_predictions_dir


# Used by API


def create_predicted_buildings_dataset(geom, checkpoint_name, data_config_path, region_name):
    skip_to_postprocess = False  # For testing

    if skip_to_postprocess == False:
        run_ml_predictions(checkpoint_name, region_name,
                           config_path=data_config_path, geom=geom)

        time.sleep(2)  # Wait for complete saving to disk

    raster_dir = get_raster_predictions_dir(region_name, checkpoint_name)
    print('Starting postprocess')
    predictions_path = sorted(
        glob.glob(raster_dir))

    crs = get_defined_crs_from_config_path(data_config_path)
    all_predicted_buildings_dataset = get_all_predicted_buildings_dataset(
        predictions_path, crs)
    return all_predicted_buildings_dataset


def create_building_dataset(geom, checkpoint_name, region_name, data_config_path, only_raw_predictions, skip_to_postprocess,
                            max_mosaic_batch_size=200, save_to='azure', num_processes=None):

    with open(data_config_path, "r") as config_file:
        config = json.load(config_file)

    if not skip_to_postprocess:
        projection = get_projection_from_config_path(data_config_path)

        run_ml_predictions(checkpoint_name, region_name, projection,
                           config_path=data_config_path, geom=geom, num_processes=num_processes)

        time.sleep(2)  # Wait for complete saving to disk

    print('Starting postprocess')

    vector_output_dir = get_vector_predictions_dir(
        region_name, checkpoint_name)
    raster_predictions_path = get_raster_predictions_dir(
        region_name, checkpoint_name)

    produce_vector_buildings(
        vector_output_dir, raster_predictions_path, config, max_mosaic_batch_size, only_raw_predictions, f"{region_name}_{checkpoint_name}", save_to)


def save_dataset(data, filename, output_dir, modelname, save_to):
    if(save_to == 'azure'):
        upload_data_to_azure(data,  modelname+'/'+filename, env.get_env_variable(
            "building_datasets_container_name"))
    else:
        save_dataset_locally(data, filename, output_dir)


def save_dataset_locally(data, filename, output_dir):
    file = open(os.path.join(
        output_dir, filename), 'w')
    file.write(data)
    file.close()


def run_ml_predictions(checkpoint_name, region_name, projection, config_path=None, geom=None,
                       skip_data_fetching=False, tupple_data=False, download_labels=False, batch_size=8,
                       save_to='local', num_processes=None):
    from kartai.tools.predict import save_predicted_images_as_geotiff

    dataset_path_to_predict = get_dataset_to_predict_dir(region_name)

    if skip_data_fetching == False:
        prepare_dataset_to_predict(region_name, geom, config_path, num_processes=num_processes)

    raster_output_dir = get_raster_predictions_dir(
        region_name, checkpoint_name)

    raster_predictions_already_exist = os.path.exists(raster_output_dir)
    if(raster_predictions_already_exist):
        print(
            f'Folder for raster predictions for {region_name} created by {checkpoint_name} already exist.')
        skip_running_prediction = input(
            "Do you want to skip predictions? Answer 'y' to skip creating new predictions, and 'n' if you want to produce new ones: ")
        if skip_running_prediction == 'y':
            return
    else:
        os.makedirs(raster_output_dir)

    model = get_ml_model(checkpoint_name)

    # Read file with references to created ortofoto images that should be analyzed
    # Prediction data is without height info, therefor crashes

    with open(dataset_path_to_predict) as f:
        prediction_input_list = Tile.tileset_from_json(json.load(f))

    img_dims = get_image_dims(prediction_input_list, tupple_data)

    batch_size = min(batch_size, len(prediction_input_list)-1)
    num_predictions = len(prediction_input_list)
    splits = (num_predictions//batch_size) if num_predictions % batch_size == 0 else (
        num_predictions//batch_size) + 1

    for i in range(splits):
        print(
            f'Run batch {i} of {splits}. Instances {batch_size*i} to {batch_size*i+batch_size}.')
        input_batch = prediction_input_list[batch_size *
                                            i:batch_size*i+batch_size]

        # Check if batch is already produced and saved:
        last_in_batch = Path(
            input_batch[-1]['image'].file_path).stem+"_prediction.tif"
        if(os.path.exists(os.path.join(raster_output_dir, last_in_batch))):
            print("batch already produced - skipping to next")
            continue

        # Generates stack of images as an array with shape (batch_size x height x length x channels)
        if tupple_data:
            tupples_to_predict = get_tuples_to_predict(input_batch)
            # If lidar images => add the lidar channel to the image to predict as an extra channel
            np_pred_results_iteration = model.predict(tupples_to_predict)

        else:
            images_to_predict = get_images_to_predict(
                input_batch, img_dims, download_labels)
            np_pred_results_iteration = model.predict(images_to_predict)

        save_predicted_images_as_geotiff(np_pred_results_iteration, input_batch,
                                         raster_output_dir, projection)


def get_dataset_to_predict_dir(region_name):
    dataset_dir = env.get_env_variable('created_datasets_directory')

    prediction_dataset_dir = os.path.join(
        dataset_dir, "for_prediction")
    if not os.path.exists(prediction_dataset_dir):
        os.makedirs(prediction_dataset_dir)

    dataset_path_to_predict = os.path.join(
        prediction_dataset_dir, region_name+".json")

    return dataset_path_to_predict


def prepare_dataset_to_predict(region_name, geom, config_path, num_processes=None):
    from kartai.dataset.PredictionArea import fetch_data_to_predict

    dataset_path_to_predict = get_dataset_to_predict_dir(region_name)

    # Create ortofoto tiles for bbox area
    if(os.path.exists(dataset_path_to_predict)):
        print(f'A dataset for area name {region_name} already exist')
        skip_dataset_fetching = input(
            "Do you want to use the previously defined dataset for this area name? Answer 'y' to skip creating dataset, and 'n' if you want to produce a new one: ")
        if not skip_dataset_fetching == 'y':
            fetch_data_to_predict(
                geom, config_path, dataset_path_to_predict, num_processes=num_processes)
    else:
        fetch_data_to_predict(geom, config_path, dataset_path_to_predict, num_processes=num_processes)


def get_ml_model(checkpoint_name):
    from tensorflow import keras
    from kartai.tools.train import getLoss
    from azure import blobstorage
    from kartai.metrics.meanIoU import (IoU, IoU_fz, Iou_point_5, Iou_point_6,
                                        Iou_point_7, Iou_point_8, Iou_point_9)

    checkpoint_path = os.path.join(env.get_env_variable(
        'trained_models_directory'), checkpoint_name)

    if(not os.path.isdir(checkpoint_path)):
        checkpoint_path = os.path.join(env.get_env_variable(
            'trained_models_directory'), checkpoint_name+".h5")

        if not os.path.isfile(checkpoint_path):
            blobstorage.download_model_file_from_azure(
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
    return model


def get_image_dims(prediction_input_list, tupple_data):
    if not tupple_data:
        if("image" in prediction_input_list[0] and "lidar" in prediction_input_list[0]):
            img_dims = [512, 512, 4]
        elif("image" in prediction_input_list[0]):
            img_dims = [512, 512, 3]
        elif("lidar" in prediction_input_list[0]):
            img_dims = [512, 512, 1]
        else:
            sys.exit("Unknown input type dimensions")
    return img_dims


def get_images_to_predict(input_batch, img_dims, download_labels):
    images_to_predict = np.empty(
        (len(input_batch), img_dims[0], img_dims[1], img_dims[2]))
    for i_batch in range(len(input_batch)):
        # Open/download image and label
        gdal_image = input_batch[i_batch]['image'].array

        if(download_labels):
            # Call code in order to download label which is needed later on
            input_batch[i_batch]['label'].array

        image = gdal_image.transpose((1, 2, 0))
        if('lidar' in input_batch[i_batch]):
            lidar = input_batch[i_batch]['lidar'].array.reshape(
                512, 512, 1)
            combined_arr = np.concatenate((image, lidar), axis=2)
            images_to_predict[i_batch, ] = combined_arr
        else:
            images_to_predict[i_batch, ] = image

    return images_to_predict


def get_tuples_to_predict(input_batch):
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

    return tupples_to_predict


def produce_vector_buildings(output_dir, raster_predictions_path, config, max_batch_size, only_raw_predictions, modelname, save_to):
    predictions_path = sorted(
        glob.glob(raster_predictions_path))
    print('output_dir', output_dir)
    print('num predictions', len(predictions_path))

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
            new_tilbygg_dataset, new_frittliggende_bygg_dataset, existing_buildings_dataset, all_predicted_buildings_dataset = create_categorised_predicted_buldings_vectordata(
                batch_prediction_paths, output_dir, config)

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
            all_predicted_buildings_dataset = create_all_predicted_buildings_vectordata(
                batch_prediction_paths, config)
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


def create_all_predicted_buildings_vectordata(predictions_path, config):
    crs = get_defined_crs_from_config(config)

    all_predicted_buildings_dataset = get_all_predicted_buildings_dataset(
        predictions_path, crs)

    if all_predicted_buildings_dataset.empty:
        return None

    # justering på datasettene

    raw_prediction_imgs = get_raw_predictions(predictions_path)
    full_img, full_transform = rasterio.merge.merge(raw_prediction_imgs)

    all_predicted_buildings_dataset = add_probability_values(
        all_predicted_buildings_dataset, full_img, full_transform, crs)

    return all_predicted_buildings_dataset.to_json()


def create_categorised_predicted_buldings_vectordata(predictions_path, output_dir, config):
    crs = get_defined_crs_from_config(config)

    all_predicted_buildings_dataset = get_all_predicted_buildings_dataset(
        predictions_path, crs)

    labels_dataset = get_labels_dataset(
        predictions_path, output_dir, config, crs)

    if labels_dataset.empty or all_predicted_buildings_dataset.empty:
        return None, None, None, None

    new_buildings_dataset = get_new_buildings_dataset(
        all_predicted_buildings_dataset, labels_dataset)

    # Setting labels area in order to filter the following datasets
    all_predicted_buildings_dataset['labels_area'] = all_predicted_buildings_dataset.geometry.area - \
        new_buildings_dataset.geometry.area

    new_frittliggende_bygg_dataset = get_new_frittliggende_dataset(
        all_predicted_buildings_dataset)

    new_tilbygg_dataset = get_tilbygg_dataset(
        all_predicted_buildings_dataset, labels_dataset)

    existing_buildings = get_existing_buildings_dataset(
        all_predicted_buildings_dataset,  new_tilbygg_dataset, new_frittliggende_bygg_dataset)

    # justering på datasettene

    raw_prediction_imgs = get_raw_predictions(predictions_path)
    full_img, full_transform = rasterio.merge.merge(raw_prediction_imgs)

    new_tilbygg_dataset = perform_last_adjustments(
        new_tilbygg_dataset, full_img, full_transform, crs)
    new_frittliggende_bygg_dataset = perform_last_adjustments(
        new_frittliggende_bygg_dataset, full_img, full_transform, crs)
    existing_buildings = perform_last_adjustments(
        existing_buildings, full_img, full_transform, crs)

    all_predicted_buildings_dataset = add_probability_values(
        all_predicted_buildings_dataset, full_img, full_transform, crs)

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


def get_labels_dataset(predictions_path, output_dir, config, crs, is_ksand_test=False):
    label_tiles = gp.GeoDataFrame()

    for prediction_path in predictions_path:
        label_path = get_label_for_prediction_path(
            prediction_path,  output_dir, config, is_ksand_test)
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


def get_label_for_prediction_path(prediction_path, output_dir, config, is_ksand_test=False):
    labels_folder = get_labels_folder(config, is_ksand_test)
    label_path = prediction_path.replace('_'+output_prediction_suffix, "").replace(
        output_dir, labels_folder)
    return label_path


def get_labels_folder(config, is_ksand_test=False):
    tilegrid = config["TileGrid"]
    cache_folder_name = f"{tilegrid['srid']}_{tilegrid['x0']}_{tilegrid['y0']}_{tilegrid['dx']}_{tilegrid['dy']}"
    labelSourceName = get_label_source_name(config, is_ksand_test)

    byggDb_path = os.path.join(env.get_env_variable(
        "cached_data_directory"), labelSourceName, cache_folder_name, "512")
    return byggDb_path


def get_label_source_name(config, is_ksand_test=False):
    labelSourceName = None
    if(is_ksand_test):
        labelSourceName = "Bygg_ksand_manuell_prosjekt"
    else:
        for source in config["ImageSources"]:
            if(source["type"] == "PostgresImageSource"):
                labelSourceName = source["name"]

    if(not labelSourceName):
        raise Exception("Could not find PostgresImageSource in config")
    return labelSourceName
