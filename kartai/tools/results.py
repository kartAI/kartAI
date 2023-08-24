
import argparse
import datetime
import glob
import json
import os
from pathlib import Path
import pandas as pd
from azure import blobstorage
import shutil
import env
from pandasgui import show
from kartai.dataset.create_polygon_dataset import get_all_predicted_features_dataset, get_fkb_labels, run_ml_predictions
from kartai.dataset.performance_count import get_new_buildings_fasit, get_performance_count_for_detected_buildings, get_true_labels
from kartai.dataset.resultRegion import ResultRegion
from kartai.dataset.test_area_utils import get_test_region_avgrensning_dir
from kartai.models.model import Model
from kartai.tools.create_training_data import create_training_data
from kartai.dataset.Iou_calculations import get_iou_for_region
from kartai.utils.prediction_utils import get_raster_predictions_dir
from kartai.utils.crs_utils import get_defined_crs_from_config_path


def add_parser(subparser):
    parser = subparser.add_parser(
        "results",
        help="show results table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-test_region', type=str, choices=ResultRegion.get_values(),
                        help='A test region to run prediction on', required=False)
    parser.add_argument('-preview', type=bool,
                        help='preview results so far', required=False, default=False)
    parser.add_argument('-download_models', type=bool,
                        help='Downloading existing trained models from azure', required=False, default=False)

    parser.set_defaults(func=main)


def download_all_performance_meta_files(region_name: ResultRegion):
    # Start by downloading all models if not already downloaded
    output_directory = get_performance_output_dir(region_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    blobstorage.download_performances(output_directory, region_name)
    performance_metafiles = blobstorage.get_available_performances(region_name)
    return performance_metafiles


def download_all_models():
    # Start by downloading all models if not already downloaded
    output_directory = env.get_env_variable('trained_models_directory')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    blobstorage.download_trained_models()
    models = blobstorage.get_available_trained_models()
    return models


def create_dataframe_result(models: list[Model], region_name: ResultRegion):
    rows = []
    for model in models:
        model_name = Path(model).stem
        performance_metadata_path = get_performance_meta_path(
            model_name, region_name)

        if (not os.path.isfile(performance_metadata_path)):
            continue
        with open(performance_metadata_path) as f:
            metadata = json.load(f)

        keys = metadata.keys()

        row = {}
        for key in keys:
            value = metadata[key]
            row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    try:
        df.sort_values('IoU', inplace=True, ascending=False)
    except:
        df.sort_values('ksand_IoU', inplace=True, ascending=False)

    print(df)
    show(df)

    df.to_excel(os.path.join(get_performance_output_dir(region_name),
                f'report_{region_name}_performance.xlsx'), index=False)


def show_general_model_performance(models):
    output_directory = env.get_env_variable('trained_models_directory')
    rows = []

    for model in models:
        model_name = Path(model).stem
        with open(os.path.join(output_directory, model_name+'.meta.json')) as f:
            metadata = json.load(f)

        keys = metadata.keys()

        row = {}
        for key in keys:
            if (key == 'training_results'):
                value = max(metadata[key]['val_Iou_point_5'])
                row['Val IoU .5'] = value
                value = max(metadata[key]['Iou_point_5'])
                row['Train IoU .5'] = value
                value = max(metadata[key]['val_Iou_point_6'])
                row['Val IoU .6'] = value
                value = max(metadata[key]['Iou_point_6'])
                row['Train IoU .6'] = value
            else:
                value = metadata[key]
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values('Val IoU .5', inplace=True, ascending=False)
    print(df)
    show(df)

    df.to_excel(os.path.join(output_directory,
                'report_trained_models.xlsx'), index=False)


def get_dataset_name_and_path(model_name: str, dataset_name: str):
    # TODO: add better check in order to find if model is height model and stack/twin model
    tupple_data = False
    if ('ndh' in model_name or 'twin' in model_name):
        if 'twin' in model_name:
            tupple_data = True
        area_dataset_name = f'{dataset_name}_ndh_prosjektomrade_not_adjusted_test_set'
    else:
        area_dataset_name = f'{dataset_name}_prosjektomrade_not_adjusted_test_set'

    area_dataset_path = os.path.join(env.get_env_variable(
        'created_datasets_directory'), area_dataset_name+'/test_set.json')

    return area_dataset_name, area_dataset_path, tupple_data


def run_performance_tests(models: list[Model], crs: str, region: str, region_name: ResultRegion, config_path: str):
    # Create performance-metadata file for each performance test

    output_predictions_name = "_prediction.tif"
    download_all_performance_meta_files(region_name)

    crs = get_defined_crs_from_config_path(config_path)

    with open(config_path, "r") as config_file:
        config: dict = json.load(config_file)

    true_labels = get_true_labels(
        region_name, region, crs)

    fkb_labels = get_fkb_labels(
        config, region, crs)

    new_buildings_fasit = get_new_buildings_fasit(
        true_labels, fkb_labels)

    new_buildings_fasit.to_file(
        "new_buildings_fasit.geojson", driver="GeoJSON", index=False)

    for model in models:
        model_name = Path(model).stem
        dataset_name, dataset_path, tupple_data = get_dataset_name_and_path(
            model_name, region_name)
        if (not os.path.exists(dataset_path)):
            create_performance_validaton_dataset(dataset_name, config_path)

        iteration = models.index(model)

        if (has_run_performance_check(model_name, region_name)):
            continue

        print(f'Start proccess for model {iteration} of {len(models)}')

        if "segformer" in model_name:
            # In order to test a segformer model you have to create the raster predictions in other repo, and then
            # copy them to the results folder. Checking if that folder exist:
            raster_output_dir = get_raster_predictions_dir(
                region_name+"_test_area", model_name)
            if not os.path.isdir(raster_output_dir):
                raise Exception(
                    f'Raster predictions for model {model_name} is not defined. Since it is a segformer model, the prediction rasters have to be produced in a different repository, and then copied to the results folder')
        else:
            run_ml_predictions(model_name, region_name+"_test_area", crs, input_model_subfolder=None, dataset_path_to_predict=dataset_path,
                               skip_data_fetching=True, tupple_data=tupple_data, download_labels=True)

        predictions_path = sorted(
            glob.glob(get_raster_predictions_dir(region_name+"_test_area", model)+f"/*{output_predictions_name}"))

        prediction_dataset_gdf = get_all_predicted_features_dataset(
            predictions_path, crs, region)

        performance_output_dir = get_performance_output_dir(region_name)

        false_count, true_count, true_new_buildings_count, all_missing_count = get_performance_count_for_detected_buildings(
            prediction_dataset_gdf, predictions_path, true_labels, new_buildings_fasit, model_name, performance_output_dir)

        iou = get_iou_for_region(
            prediction_dataset_gdf, true_labels, region, crs)
        print('False detected buildings:', false_count)
        print('True detected buildings:', true_count)
        print('All missing buildings', all_missing_count)

        create_performance_metadata_file(region_name,
                                         iou, model_name, false_count, true_count, true_new_buildings_count, all_missing_count)

        blobstorage.upload_model_performance_file(
            model_name + f'_{region_name}_performance', region_name)

    create_dataframe_result(models, region_name)


def create_performance_validaton_dataset(dataset_name: str, config_path: str):
    """Creating dataset for the region to make resulttable for"""
    if ResultRegion.KRISTIANSAND == ResultRegion.from_str(dataset_name):
        create_training_data(dataset_name,
                             config_path, eager_load=True, x_min=437300, x_max=445700, y_min=6442000, y_max=6447400)
    elif ResultRegion.BALSFJORD == ResultRegion.from_str(dataset_name):
        create_training_data(dataset_name,
                             config_path, eager_load=True, region="training_data/regions/balsfjord_test_area.geojson")
    else:
        raise NotImplementedError("Unsupported region")


def has_run_performance_check(model_name: str, region_name: str):
    """Checks wether there already exists a peformance file for the given model in the given region"""
    meta_path = get_performance_meta_path(model_name, region_name)
    return os.path.isfile(meta_path)


def get_performance_output_dir(name: str):
    """Directory for where performance files should be saved"""
    return os.path.join(env.get_env_variable(
        'prediction_results_directory'), name+"_performance")


def get_predictions_output_dir():
    """Directory for where prediction files should be saved when running "result" module """
    return os.path.join(env.get_env_variable(
        'prediction_results_directory'), 'prediction_for_performance_test')


def get_performance_meta_path(model_name: str, region_name: str):
    """Directory for where performance meta file should be saved"""
    out_folder = get_performance_output_dir(region_name)
    file_name = model_name + f'_{region_name}_performance.json'
    path = os.path.join(out_folder, file_name)
    return path


def get_checkpoint_meta_file_dir(model_name: Model):
    checkpoints_directory = env.get_env_variable('trained_models_directory')

    # Support checkpoints from old and new format
    kartai_dir = os.path.join(checkpoints_directory, model_name+'.meta.json')
    kartai_stream_dir = os.path.join(os.path.join(
        checkpoints_directory, model_name), model_name+'_meta.json')

    if os.path.exists(kartai_dir):
        return kartai_dir
    elif os.path.exists(kartai_stream_dir):
        return kartai_stream_dir
    else:
        raise Exception("Cannot find meta file for model:", model_name)


def create_performance_metadata_file(region_name: ResultRegion, IoU: float, model_name: str, false_count: int, true_buildings_count: int, true_new_buildings_count: int, all_missing_count: int):

    out_folder = get_performance_output_dir(region_name)

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    meta_dir = get_checkpoint_meta_file_dir(model_name)
    with open(meta_dir) as f:
        training_metadata_file = json.load(f)

    ct = datetime.datetime.now()

    meta = {
        "model_name": model_name,
        "region": region_name,
        "IoU": IoU,
        "Sanne detekterte bygnigner": true_buildings_count,
        "Falske detekterte bygninger": false_count,
        "Prosentandel detektert bygg": (true_buildings_count / (true_buildings_count + all_missing_count))*100,
        "Manglende detekterte bygninger": all_missing_count,
        "Sanne detekterte 'nye' bygninger": true_new_buildings_count,
        "training_params": {
            "val_iou_point_5": get_training_iou_results(training_metadata_file),
            "dataset": training_metadata_file['training_dataset_name'] if "training_dataset_name" in training_metadata_file else "kartai-stream",
        },
        "date_time": str(ct),
    }
    ident = 2

    performance_file = get_performance_meta_path(model_name, region_name)

    with open(performance_file, 'w') as outfile:
        json.dump(meta, outfile, indent=ident)

    print("created metadata-file:\n", json.dumps(meta,  indent=ident))


def get_training_iou_results(training_metadata_file: object):
    """Get IoU metric from the metadata file from a trained model"""
    training_results = -1
    if ("training_results" in training_metadata_file):
        return max(
            training_metadata_file['training_results']['val_Iou_point_5'])
    else:
        for log in training_metadata_file["logs"]:
            if "val_segformer_iou_point_5" in log:
                if log["val_segformer_iou_point_5"] > training_results:
                    return log["val_segformer_iou_point_5"]
            elif "val_Iou_point_5" in log:
                if log["val_Iou_point_5"] > training_results:
                    return log["val_Iou_point_5"]
            elif "val_IoU" in log:
                if log["val_IoU"] > training_results:
                    return log["val_IoU"]
            else:
                raise NotImplementedError("Unknown metric")


def empty_folder(folder: str):
    if (not os.path.exists(folder)):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def main(args):

    if args.download_models:
        download_data = input(
            "Do you want to download model checkpoints from azure? \nDownload? Answer 'y' \nSkip? Answer 'n':\n ")
        if download_data == 'y':
            download_all_models()

    checkpoints_directory = env.get_env_variable('trained_models_directory')

    checkpoint_files = glob.glob(checkpoints_directory + "/*.h5")

    local_models_kartai = []
    for checkpoint_path in checkpoint_files:
        local_models_kartai.append(Path(Path(checkpoint_path).name).stem)

    local_models_kartai_stream = [name for name in os.listdir(
        checkpoints_directory) if os.path.isdir(os.path.join(checkpoints_directory, name))]

    models = local_models_kartai + local_models_kartai_stream

    crs = "EPSG:25832"
    if args.preview is True:
        create_dataframe_result(
            models, region_name=args.test_region)
    elif args.test_region:
        region = get_test_region_avgrensning_dir(args.test_region)
        run_performance_tests(models, crs, region, region_name=args.test_region,
                              config_path="config/dataset/bygg_test.json")
    else:
        show_general_model_performance(models)
