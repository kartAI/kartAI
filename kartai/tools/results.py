
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
from kartai.dataset.create_building_dataset import get_all_predicted_buildings_dataset, run_ml_predictions
from kartai.dataset.performance_count import get_performance_count_for_detected_buildings
from kartai.tools.create_training_data import create_training_data
from kartai.dataset.Iou_calculations import get_IoU_for_ksand


def add_parser(subparser):
    parser = subparser.add_parser(
        "results",
        help="show results table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-ksand', type=bool,
                        help='Whether to run ksand tests for the models as well', required=False)
    parser.add_argument('-preview', type=bool,
                        help='preview results so far', required=False)

    parser.set_defaults(func=main)


def download_all_ksand_performance_meta_files():
    # Start by downloading all models if not already downloaded
    output_directory = get_ksand_performance_output_dir()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    blobstorage.download_ksand_performances(output_directory)
    performance_metafiles = blobstorage.get_available_ksand_performances()
    return performance_metafiles


def download_all_models():
    # Start by downloading all models if not already downloaded
    output_directory = env.get_env_variable('trained_models_directory')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    blobstorage.downloadTrainedModels()
    models = blobstorage.getAvailableTrainedModels()
    return models


def create_ksand_dataframe_result(models):
    rows = []

    for model in models:
        model_name = Path(model).stem
        performance_metadata_path = get_ksand_performance_meta_path(model_name)
        if(not os.path.isfile(performance_metadata_path)):
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
    df.sort_values('ksand_IoU', inplace=True, ascending=False)
    print(df)
    show(df)

    df.to_excel(os.path.join(get_ksand_performance_output_dir(),
                'report_ksand_performance.xlsx'), index=False)


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
            if(key == 'training_results'):
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


def get_ksand_dataset_name_and_path(model_name):
    # TODO: add better check for in order to find if model is height model and stack/twin model
    tupple_data = False
    if('ndh' in model_name or 'twin' in model_name):
        if 'twin' in model_name:
            tupple_data = True
        ksand_dataset_name = 'ksand_ndh_prosjektomrade_not_adjusted_test_set'
    else:
        ksand_dataset_name = 'ksand_prosjektomrade_not_adjusted_test_set'

    ksand_dataset_path = os.path.join(env.get_env_variable(
        'created_datasets_directory'), ksand_dataset_name+'/test_set.json')

    return ksand_dataset_name, ksand_dataset_path, tupple_data


def run_ksand_tests(models):
    # Create performance-metadata file for each performance test

    output_predictions_name = "_prediction.tif"
    #performance_metafiles = download_all_ksand_performance_meta_files()
    predictions_output_dir = get_predictions_output_dir()

    for model in models:
        model_name = Path(model).stem
        ksand_dataset_name, ksand_dataset_path, tupple_data = get_ksand_dataset_name_and_path(
            model_name)
        if(not os.path.isfile(ksand_dataset_path)):
            create_ksand_validaton_dataset(ksand_dataset_name)

        iteration = models.index(model)

        if(has_run_performance_check(model_name)):
            continue

        print(f'Start proccess for model {iteration} of {len(models)}')
        # Clean current content of folder to make sure only current batch is in folder
        empty_folder(predictions_output_dir)
        run_ml_predictions(model_name, predictions_output_dir, output_predictions_name,
                           skip_data_fetching=True, dataset_path_to_predict=ksand_dataset_path, tupple_data=tupple_data)

        predictions_path = sorted(
            glob.glob(predictions_output_dir+f"/*{output_predictions_name}"))

        prediction_dataset_gdf = get_all_predicted_buildings_dataset(
            predictions_path)

        IoU_ksand = get_IoU_for_ksand(prediction_dataset_gdf)

        performance_output_dir = get_ksand_performance_output_dir()

        false_count, true_count, true_new_buildings_count, fkb_missing_count, all_missing_count = get_performance_count_for_detected_buildings(
            prediction_dataset_gdf, predictions_path, model_name, predictions_output_dir, performance_output_dir)

        print('False detected buildings:', false_count)
        print('True detected buildings:', true_count)
        print('Missed new building (not in fkb):', fkb_missing_count)
        print('All missing buildings', all_missing_count)

        create_ksand_performance_metadata_file(
            IoU_ksand, model_name, false_count, true_count, true_new_buildings_count, fkb_missing_count, all_missing_count)

        blobstorage.upload_ksand_model_performance_file(
            model_name + '_ksand_performance')

    create_ksand_dataframe_result(models)


def create_ksand_validaton_dataset(ksand_dataset_name):
    config_path = 'config/dataset/ksand-manuell.json'
    create_training_data(ksand_dataset_name,
                         config_path,  x_min=437300, x_max=445700, y_min=6442000, y_max=6447400)


def has_run_performance_check(model_name):
    meta_path = get_ksand_performance_meta_path(model_name)
    return os.path.isfile(meta_path)


def get_ksand_performance_output_dir():
    return os.path.join(env.get_env_variable(
        'prediction_results_directory'), 'ksand_performance')


def get_predictions_output_dir():
    return os.path.join(env.get_env_variable(
        'prediction_results_directory'), 'prediction_for_performance_test')


def get_ksand_performance_meta_path(model_name):
    out_folder = get_ksand_performance_output_dir()
    file_name = model_name + '_ksand_performance.json'
    path = os.path.join(out_folder, file_name)
    return path


def create_ksand_performance_metadata_file(ksand_IoU, model_name, false_count, true_buildings_count, true_new_buildings_count, fkb_missing_count, all_missing_count):

    checkpoints_directory = env.get_env_variable('trained_models_directory')
    out_folder = get_ksand_performance_output_dir()

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    with open(os.path.join(checkpoints_directory, model_name+'.meta.json')) as f:
        training_metadata_file = json.load(f)

    ct = datetime.datetime.now()

    meta = {
        "model_name": model_name,
        "ksand_IoU": ksand_IoU,
        "Sanne detekterte bygnigner": true_buildings_count,
        "Sanne detekterte 'nye' bygninger": true_new_buildings_count,
        "Falske detekterte bygninger": false_count,
        "Manglende detekterte bygninger": all_missing_count,
        "Manglende detekterte 'nye' bygninger": fkb_missing_count,
        "training_params": {
            "val_iou_point_5": max(training_metadata_file['training_results']['val_Iou_point_5']),
            "dataset": training_metadata_file['training_dataset_name'],
        },
        "date_time": str(ct),
    }
    ident = 2

    file = get_ksand_performance_meta_path(model_name)

    with open(file, 'w') as outfile:
        json.dump(meta, outfile, indent=ident)

    print("created metadata-file:\n", json.dumps(meta,  indent=ident))


def empty_folder(folder):
    if(not os.path.exists(folder)):
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

    models = download_all_models()

    if args.ksand == True:
        run_ksand_tests(models)
    elif args.preview == True:
        create_ksand_dataframe_result(models)
    else:
        show_general_model_performance(models)
