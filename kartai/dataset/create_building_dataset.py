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
from tensorflow import keras
import tensorflow as tf
import env
import rasterio
import rasterio.features
import rasterio.merge
from rasterstats import zonal_stats
from kartai.datamodels_and_services.ImageSourceServices import Tile
from kartai.utils.confidence import Confidence
from kartai.utils.crs_utils import get_defined_crs_from_config, get_defined_crs_from_config_path, get_projection_from_config_path
from kartai.tools.predict import save_predicted_images_as_geotiff
from kartai.utils.dataset_utils import get_X_tuple
from kartai.utils.geometry_utils import parse_region_arg
from kartai.utils.prediction_utils import get_raster_predictions_dir, get_vector_predictions_dir
from kartai.tools.train import getLoss
from kartai.metrics.meanIoU import (IoU, IoU_fz, Iou_point_5, Iou_point_6,
                                    Iou_point_7, Iou_point_8, Iou_point_9)
from sqlalchemy import create_engine

# Used by API


def create_predicted_buildings_dataset(geom, checkpoint_name, data_config_path, region_name):
    skip_to_postprocess = False  # For testing
    projection = "EPSG:25832"
    if skip_to_postprocess == False:
        if "segformer" in checkpoint_name:
            # In order to test a segformer model you have to create the raster predictions in other repo, and then
            # copy them to the results folder. Checking if that folder exist:
            raster_output_dir = get_raster_predictions_dir(
                region_name, checkpoint_name)
            if not os.path.isdir(raster_output_dir):
                raise Exception(
                    f'Raster predictions for model {checkpoint_name} is not defined. Since it is a segformer model, the prediction rasters have to be produced in a different repository, and then copied to the results folder')
        else:
            run_ml_predictions(checkpoint_name, region_name, projection=projection,
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


def create_building_dataset(geom, checkpoint_name, region_name, data_config_path, skip_to_postprocess,
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
        vector_output_dir, raster_predictions_path, config, max_mosaic_batch_size, f"{region_name}_{checkpoint_name}", save_to)


def save_dataset(data, filename, output_dir, modelname, save_to):
    if (save_to == 'azure'):
        upload_data_to_azure(data,  modelname+'/'+filename, env.get_env_variable(
            "building_datasets_container_name"))
    else:
        save_dataset_locally(data, filename, output_dir)


def save_dataset_locally(data, filename, output_dir):
    file = open(os.path.join(
        output_dir, filename), 'w')
    file.write(data)
    file.close()


def run_ml_predictions(input_model_name, region_name, projection, input_model_subfolder=None, dataset_path_to_predict=None, config_path=None, geom=None,
                       skip_data_fetching=False, tupple_data=False, download_labels=False, batch_size=8,
                       save_to='local', num_processes=None):
    """Running prediction on a dataset containing tiled input data, or a region to created data for """

    print("\nRunning ML prediction")
    dataset_path_to_predict = dataset_path_to_predict if dataset_path_to_predict else get_dataset_to_predict_dir(
        region_name)

    if skip_data_fetching == False:
        prepare_dataset_to_predict(
            region_name, geom, config_path, num_processes=num_processes)

    raster_output_dir = get_raster_predictions_dir(
        region_name, input_model_name)

    raster_predictions_dir_already_exist = os.path.exists(raster_output_dir)
    if (not raster_predictions_dir_already_exist):
        os.makedirs(raster_output_dir)

    model = get_ml_model(input_model_name, input_model_subfolder)

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
        if (os.path.exists(os.path.join(raster_output_dir, last_in_batch))):
            print("batch already produced - skipping to next")
            continue

        # Generates stack of images as an array with shape (batch_size x height x length x channels)
        if tupple_data:
            tupples_to_predict = get_tuples_to_predict(input_batch)
            # If lidar images => add the lidar channel to the image to predict as an extra channel
            np_pred_results_iteration = predict(model, tupples_to_predict)

        else:
            images_to_predict = get_images_to_predict(
                input_batch, img_dims, download_labels)
            np_pred_results_iteration = predict(model, images_to_predict)

        save_predicted_images_as_geotiff(np_pred_results_iteration, input_batch,
                                         raster_output_dir, projection)


def predict(model, images_to_predict):
    return model(tf.convert_to_tensor(images_to_predict), training=False).numpy()


def get_dataset_to_predict_dir(region_name, suffix=None):
    dataset_dir = env.get_env_variable('created_datasets_directory')

    prediction_dataset_dir = os.path.join(
        dataset_dir, "for_prediction")
    if not os.path.exists(prediction_dataset_dir):
        os.makedirs(prediction_dataset_dir)

    dataset_name = region_name+suffix+".json" if suffix else region_name+".json"
    dataset_path_to_predict = os.path.join(
        prediction_dataset_dir, dataset_name)

    return dataset_path_to_predict


def prepare_dataset_to_predict(region_name, geom, config_path, num_processes=None):
    from kartai.dataset.PredictionArea import fetch_data_to_predict

    dataset_path_to_predict = get_dataset_to_predict_dir(region_name)

    # Create ortofoto tiles for bbox area
    if (os.path.exists(dataset_path_to_predict)):
        skip_dataset_fetching = input(
            f"A dataset for area name {region_name} already exist. You can either use existing dataset by skipping step, or create a new. \nSkip? Answer 'y' \nCreate new? Answer 'n':\n ")
        if not skip_dataset_fetching == 'y':
            fetch_data_to_predict(
                geom, config_path, dataset_path_to_predict, num_processes=num_processes)
    else:
        fetch_data_to_predict(
            geom, config_path, dataset_path_to_predict, num_processes=num_processes)


def get_ml_model(input_model_name, input_model_subfolder=None):

    checkpoint_path = get_checkpoint_path(
        input_model_name, input_model_subfolder)

    dependencies = {
        'BinaryFocalLoss': getLoss('focal_loss'),
        'Iou_point_5': Iou_point_5,
        'Iou_point_6': Iou_point_6,
        'Iou_point_7': Iou_point_7,
        'Iou_point_8': Iou_point_8,
        'Iou_point_9': Iou_point_9,
        'IoU': IoU,
        'IoU_fz': IoU_fz,
        "Confidence": Confidence()
    }
    model = keras.models.load_model(
        checkpoint_path, custom_objects=dependencies)
    return model


def get_checkpoint_path(input_model_name, input_model_subfolder):
    """Support fetching checkpoints from three different formats:
      1: .h5 files saved directly to checkpoints folder
      2: New keras checkpoints format saved to a model directory
      3: subfolders containing the epoch and iou value area created with checkpoints files inside. This allows us to save different versions of the models
    """

    input_model_path = os.path.join(env.get_env_variable(
        "trained_models_directory"), input_model_name)

    if input_model_subfolder:
        return os.path.join(
            input_model_path, input_model_subfolder)

    if not os.path.isdir(input_model_path):
        return input_model_path+'.h5'

    existing_subfolders = os.listdir(input_model_path)
    sub_dirs = []
    for subfolder in existing_subfolders:
        if "epoch" in subfolder:
            sub_dirs.append(subfolder)

    if len(sub_dirs) == 0:
        return input_model_path
    else:
        best_metric = 0
        best_checkpoint_path = ""
        for sub_dir in sub_dirs:
            metric_in_sub_dir = get_iou_from_pathname(sub_dir)
            if metric_in_sub_dir > best_metric:
                best_metric = metric_in_sub_dir
                best_checkpoint_path = sub_dir

        return os.path.join(input_model_path, best_checkpoint_path)


def get_iou_from_pathname(path):
    """Get IoU value in path name"""
    if "IoU" not in path:
        raise Exception(
            "Sorry - Only supports fetching best checkpoint results when IoU metric is used and this checkpoint path has no IoU in its name")
    return float(path.split("-IoU_")[1])


def get_image_dims(prediction_input_list, tupple_data):
    if not tupple_data:
        if ("image" in prediction_input_list[0] and "lidar" in prediction_input_list[0]):
            img_dims = [512, 512, 4]
        elif ("image" in prediction_input_list[0]):
            img_dims = [512, 512, 3]
        elif ("lidar" in prediction_input_list[0]):
            img_dims = [512, 512, 1]
        else:
            sys.exit("Unknown input type dimensions")
    return img_dims


def get_images_to_predict(input_batch, img_dims, download_labels):
    images_to_predict = np.empty(
        (len(input_batch), img_dims[0], img_dims[1], img_dims[2]))
    for index, sample in enumerate(input_batch, start=0):
        # Open/download image and label
        gdal_image = sample['image'].array

        if (download_labels):
            # Call code in order to download label which is needed later on
            sample['label'].array

        image = gdal_image.transpose((1, 2, 0))
        if ('lidar' in sample):
            lidar = sample['lidar'].array.reshape(
                512, 512, 1)
            combined_arr = np.concatenate((image, lidar), axis=2)
            images_to_predict[index, ] = combined_arr
        else:
            images_to_predict[index, ] = image

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


def produce_vector_buildings(output_dir, raster_predictions_path, config, max_batch_size, modelname, save_to):
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
        if (i == num_splits):
            # Last run, batch size might be lower
            batch_prediction_paths = predictions_path[i *
                                                      batch_size:(len(predictions_path)-1)]
        else:
            batch_prediction_paths = predictions_path[i *
                                                      batch_size:i*batch_size+batch_size]

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
    """Create vectors of the mask"""
    shapes = rasterio.features.shapes(
        mask, connectivity=4, transform=img.transform)

    records = [{"geometry": geometry, "properties": {"value": value}}
               for (geometry, value) in shapes if value == 1]
    geoms = list(records)

    # Return empty geodataframe if there is no geometry within the mask/tile
    polygons = gp.GeoDataFrame.from_features(
        geoms, crs=crs) if len(geoms) > 0 else gp.GeoDataFrame()
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

    all_predicted_buildings_dataset = add_confidence_values(
        all_predicted_buildings_dataset, predictions_path)

    return all_predicted_buildings_dataset.to_json()


def add_confidence_values(dataset, predictions_path):
    """Adding confidence values from prediction to the vector data"""
    raw_prediction_imgs = get_raw_predictions(predictions_path)
    full_img, full_transform = rasterio.merge.merge(raw_prediction_imgs)

    dataset = dataset.reset_index(drop=True)
    probabilites = zonal_stats(dataset.geometry,
                               full_img[0], affine=full_transform, stats='mean', nodata=0)

    dataset['prob'] = gp.GeoDataFrame(probabilites)
    return dataset


def simplify_dataset(dataset):
    """Simplify the data by removing small areas, and removing points from the polygons by running simplify"""
    # Remove all small buildings in dataset
    dataset = dataset.loc[dataset.geometry.area > 1]

    # Remove points in polygons to get better building-look
    dataset['geometry'] = dataset.simplify(
        tolerance=0.25)

    return dataset


def get_fkb_labels(config, region_path, crs):
    """Get labels from FKB data"""
    layer_spec = None
    for source_config in config["ImageSources"]:
        if source_config["type"] == "PostgresImageSource":
            layer_spec = source_config

    connectionPwd = env.get_env_variable(layer_spec['passwd'])

    region_geojson_string = parse_region_arg(region_path, "text")

    # Tables need to have quotes around each table/subtable name
    subtables = layer_spec["table"].split(".")  # Splitting subtables
    table_string = ""
    for index, table in enumerate(subtables):
        last_item = index == len(subtables)-1
        table_string += f'"{table}".' if not last_item else f'"{table}"'

    sql = f"""
    SELECT st_transform(geom, {layer_spec["srid"]}) as geom
    FROM {table_string} n
    WHERE ST_Intersects(geom::geometry, st_transform(st_setsrid(ST_geomfromgeojson('{region_geojson_string}'),{layer_spec["srid"]}), 4326))
    """

    db_connection_url = f"postgresql://{layer_spec['user']}:{connectionPwd}@{layer_spec['host']}:{layer_spec['port']}/{layer_spec['database']}"
    con = create_engine(db_connection_url)

    df = gp.GeoDataFrame.from_postgis(sql, con)
    df = df.set_crs(f"EPSG:{layer_spec['srid']}")
    df = merge_connected_geoms(df)
    df = df[df.geom_type != 'Point']
    df = df[df.geom_type != 'LineString']
    df = df[df.geom_type != 'MultiLineString']

    df = clip_to_polygon(df, region_path, crs)

    return df


def get_all_predicted_buildings_dataset(predictions_path, crs, region_dir=None):
    """Create a vector dataset of the prediction raster tiles"""

    print("\nCreating vectordata from prediction tiles")
    predictions = gp.GeoDataFrame()

    for prediction_path in predictions_path:
        raw_prediction_img = rasterio.open(
            prediction_path)

        prediction_mask = create_normalized_prediction_mask(raw_prediction_img)
        prediction_polygons = polygonize_mask(
            prediction_mask, raw_prediction_img, crs)

        predictions = pd.concat(
            [predictions, prediction_polygons], ignore_index=True)

    if predictions.empty:
        return predictions

    # Remove all buildings with area less than 1 square meter
    predictions['geometry'] = get_valid_geoms(predictions)
    merged_predictions = merge_connected_geoms(predictions)

    # Remove points in polygons to get better building-look
    merged_predictions['geometry'] = merged_predictions.simplify(
        tolerance=0.25)

    # Remove all buildings with area less than 1 square meter
    cleaned_dataset = merged_predictions.loc[merged_predictions.geometry.area > 1]

    cleaned_dataset.set_crs(crs, inplace=True)

    if region_dir:
        cleaned_dataset = clip_to_polygon(cleaned_dataset, region_dir, crs)

    return cleaned_dataset


def clip_to_polygon(dataset, polygon_file, crs):
    """Clip a dataset to a polygon"""
    area_gdf = gp.read_file(polygon_file)
    area_gdf.set_crs(crs, allow_override=True, inplace=True)
    dataset = gp.clip(
        dataset, area_gdf)
    return dataset


def merge_connected_geoms(geoms):
    try:
        dissolved_geoms = geoms.dissolve()
        dissolved_geoms = dissolved_geoms.explode().reset_index(drop=True)
        return dissolved_geoms
    except Exception:
        print("could not connect geoms")
        return geoms


def get_valid_geoms(geoms):
    # Hack to clean up geoms
    if geoms.empty:
        return geoms
    return geoms.geometry.buffer(0)


def get_label_source_name(config, region_name=None, is_performance_test=False):

    labelSourceName = None
    if is_performance_test and region_name == "ksand":
        labelSourceName = "Bygg_ksand_manuell_prosjekt"
    else:
        for source in config["ImageSources"]:
            if (source["type"] == "PostgresImageSource"):
                labelSourceName = source["name"]

    if (not labelSourceName):
        raise Exception("Could not find PostgresImageSource in config")
    return labelSourceName
