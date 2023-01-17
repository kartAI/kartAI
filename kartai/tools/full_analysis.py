
import argparse
import os
import env
from kartai.models import segmentation_models
import time
import json
from kartai.utils.geometry_utils import parse_region_arg
from kartai.tools.create_training_data import create_training_data
from kartai.tools.data_teacher import run_data_teacher
from kartai.dataset.create_building_dataset import create_building_dataset
from kartai.tools.train import train


def create_validation_dataset(geom_path, validation_dataset_name):
    dataset_path = os.path.join(env.get_env_variable(
        'created_datasets_directory'), validation_dataset_name)
    if not os.path.isdir(dataset_path):
        create_training_data(
            validation_dataset_name, "config/dataset/bygg_validation.json", region=geom_path)
        time.sleep(1)  # Complete save
    else:
        print(
            f'Skipping producing validation data, dataset with name {validation_dataset_name} already exist.')


def create_init_building_dataset(geom_path, init_training_dataset_name, in_test_mode):
    dataset_path = os.path.join(env.get_env_variable(
        'created_datasets_directory'), init_training_dataset_name)

    if not os.path.isdir(dataset_path):
        dataset_config_path = "config/dataset/bygg_testmode.json" if in_test_mode else "config/dataset/bygg.json"
        create_training_data(init_training_dataset_name,
                             dataset_config_path, region=geom_path)
        time.sleep(1)  # Complete save

    else:
        print(
            f'Skipping producing init training data, dataset with name {init_training_dataset_name} already exist.')


def create_ksand_manuell_dataset(dataset_name):
    # Check if dataset already exist - if so do not create
    dataset_path = os.path.join(env.get_env_variable(
        'created_datasets_directory'), dataset_name)
    if not os.path.isdir(dataset_path):
        create_training_data(dataset_name,
                             "config/dataset/ksand-manuell.json",  x_min=437300, x_max=445700, y_min=6442000, y_max=6447400)
        time.sleep(1)  # Complete save
    else:
        print(
            f'Skipping producing ksand data, dataset with name {dataset_name} already exist.')


def start_data_teacher(name, validation_dataset_name, init_training_dataset_name, region_filepath, train_args, in_test_mode):

    input_generator_config_path = "config/ml_input_generator/ortofoto.json"
    dataset_config_file = "config/dataset/bygg_auto_expanding_testmode.json" if in_test_mode else "config/dataset/bygg_auto_expanding.json"
    init_checkpoint = 'unet_large_building_area_d4_swish' if in_test_mode else None
    init_threshold = 0.98 if in_test_mode else 0.8

    # data_teacher_name: str, validation_dataset_name: str, training_dataset_name: str, input_generator_config_path: str, dataset_config_file: str, region: str, train_args: object, init_checkpoint: str = None, init_threshold: float = None, in_test_mode: bool = False
    best_model_name = run_data_teacher(name, validation_dataset_name, init_training_dataset_name, input_generator_config_path, dataset_config_file,
                                       region_filepath, train_args, init_checkpoint=init_checkpoint, init_threshold=init_threshold, in_test_mode=in_test_mode)

    return best_model_name


def produce_building_dataset(analysis_name, trained_model_name, max_mosaic_batch_size):
    ksand_test_region_path = "training_data/regions/prosjektomr_test.json"

    geom = parse_region_arg(ksand_test_region_path)

    output_dir = os.path.join(env.get_env_variable(
        "prediction_results_directory"), analysis_name)

    config_path = "config/dataset/bygg-no-rules.json"
    only_raw_predictions = True
    skip_to_postprocess = False
    create_building_dataset(geom, trained_model_name, 'full-analysis',
                            config_path, only_raw_predictions, skip_to_postprocess, output_dir, max_mosaic_batch_size, save_to='local')


def finetune_model_on_ksand(ksand_manuell_dataset_name, model_to_finetune, new_model_name, train_args, in_test_mode):

    input_generator_config_path = "config/ml_input_generator/ortofoto.json"

    save_model = False if in_test_mode else True

    train(new_model_name, ksand_manuell_dataset_name,
          input_generator_config_path, save_model, train_args, model_to_finetune)


def get_best_model_name_from_data_teacher(analysis_name):
    metadata_name = analysis_name + '_datateacher_session.meta.json'
    metadata_path = os.path.join(env.get_env_variable(
        "trained_models_directory"), metadata_name)

    with open(metadata_path, encoding="utf8") as metadata:
        datateacher_metadata = json.load(metadata)

    return datateacher_metadata["best_model_name"]


def add_parser(subparser):
    parser = subparser.add_parser(
        "full_analysis",
        help="Run a full analysis. Creating dataset, training model and expanding dataset with datateacher, and then creating building dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--analysis_name", type=str,
                        help="Name for the analysis to run",
                        required=True)
    parser.add_argument('-m', '--model', type=str,
                        help='The wanted neural net model', choices=segmentation_models.models, required=True)
    parser.add_argument("--region_init", type=str,
                        help="Polygon boundary of init training area for data teacher\n"
                        "WKT, json text or filename\n"
                        "alternative to bounding box",
                        required=True)
    parser.add_argument("--region_validation", type=str,
                        help="Polygon boundary of area to create validation data for data teacher\n"
                        "WKT, json text or filename\n"
                        "alternative to bounding box",
                        required=True)
    parser.add_argument("--region_expand", type=str,
                        help="Polygon boundary of area that data teacher can find new data\n"
                        "WKT, json text or filename\n"
                        "alternative to bounding box",
                        required=True)
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Size of minibatch', default=8)
    parser.add_argument('-f', '--features', type=int,
                        help='Number of features in first layers', default=32)
    parser.add_argument('-d', '--depth', type=int,
                        help='Depth of U', default=4)
    parser.add_argument('-l', '--loss', type=str,
                        choices=segmentation_models.loss_functions, default='binary_crossentropy')
    parser.add_argument('-a', '--activation', type=str,
                        help='Activation function', choices=segmentation_models.activations, default='relu')
    parser.add_argument('-opt', '--optimizer', type=str,
                        help='Optimizer function', default='RMSprop')
    parser.add_argument('-test', '--in_test_mode', type=bool,
                        help='If running test of full analysis', default=False)

    parser.set_defaults(func=main)


def main(args):
    # FOR NOW ONLY SUPPORTS ORTOFOTO, NOT LASER DATA
    train_args = {
        "features": args.features,
        "depth": args.depth,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "model": args.model,
        "epochs": 1 if args.in_test_mode else 100,
        "activation": args.activation,
        "loss": args.loss
    }

    ksand_manuell_dataset_name = "kristiansand_manually_adjusted"

    create_ksand_manuell_dataset(ksand_manuell_dataset_name)

    init_training_dataset_name = "init_training_data_"+args.analysis_name
    create_init_building_dataset(
        args.region_init, init_training_dataset_name, in_test_mode=args.in_test_mode)

    validation_dataset_name = "validation_data_"+args.analysis_name
    create_validation_dataset(args.region_validation, validation_dataset_name)

    # run data_teacher
    best_model = start_data_teacher(args.analysis_name, validation_dataset_name,
                                    init_training_dataset_name, args.region_expand, train_args, in_test_mode=args.in_test_mode)

    # Finetune model on ksand data
    finetuned_model_name = f"finetuned_ksand_{args.analysis_name}"
    finetune_model_on_ksand(ksand_manuell_dataset_name,
                            best_model, finetuned_model_name, train_args, args.in_test_mode)

    max_mosaic_batch_size = 200

    # Produce building detection model
    produce_building_dataset(
        args.analysis_name, finetuned_model_name, max_mosaic_batch_size)
