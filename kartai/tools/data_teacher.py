import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from azure import blobstorage

import env
from kartai.exceptions import CheckpointNotFoundException, InvalidCheckpointException
from kartai.models import segmentation_models
from kartai.tools.create_training_data import create_training_data
from kartai.tools.predict import predict_and_evaluate
from kartai.tools.train import create_metadata_file, train_model
from kartai.utils.train_utils import check_for_existing_model, get_dataset_dirs


def _train_model(checkpoint_name, created_datasets_dirs, dataset_names, input_generator_config, input_generator_config_path, train_args):
    check_for_existing_model(checkpoint_name)

    train_history = train_model(created_datasets_dirs, input_generator_config, checkpoint_name, train_args["model"], train_args["features"],
                                train_args["depth"], train_args["activation"], train_args["batch_size"], train_args["epochs"], train_args["optimizer"],  train_args['loss'])

    create_metadata_file(created_datasets_dirs, dataset_names, Path(input_generator_config_path).stem, checkpoint_name, train_args["model"], train_args["features"],
                         train_args["depth"], train_args["activation"], train_args["batch_size"], train_args["epochs"], train_history, train_args["optimizer"],  train_args['loss'])


def generate_low_confidence_training_datasets(dataset_name, config_file_path, region, confidence_threshold, eval_model_checkpoint):
    # run create dataset
    return create_training_data(dataset_name, config_file_path, confidence_threshold=confidence_threshold, eval_model_checkpoint=eval_model_checkpoint, region=region)


def evaluate_model(checkpoint_name, evaluation_dataset_path, datagenerator_config):

    model_evaluation = predict_and_evaluate(evaluation_dataset_path,
                                            datagenerator_config, checkpoint_name, save_prediction_images=False, save_diff_images=False, generate_metadata=True, dataset_to_evaluate="valid")
    return model_evaluation


def create_datateacher_metadata_file(best_model_name, model_evaluations, models_tested, name, args):
    ct = datetime.datetime.now()

    meta = {
        "best_model_name": best_model_name,
        "model_evaluations": model_evaluations,
        "models_tested": models_tested,
        "date_time": str(ct),
        "args": args
    }

    print("meta", meta)

    ident = 2
    file = os.path.join(env.get_env_variable(
        'trained_models_directory'), name+'_datateacher_session.meta.json')

    with open(file, 'w') as outfile:
        json.dump(meta, outfile, indent=ident)

    print(json.dumps(meta,  indent=ident))


def add_parser(subparser):
    parser = subparser.add_parser(
        "data_teacher",
        help="Training of model with a 'data teacher' that continuously adds data to the training dataset that the model has low confidence on to improve model further.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-n', '--data_teacher_name', type=str,
                        help='Name to add as prefix in resulting datasets and trained models', required=True)
    parser.add_argument('-cn', '--init_checkpoint', type=str,
                        help='Checkpoint to use for first iteration. Replaces init training.', required=False)
    parser.add_argument('-v_dn', '--validation_dataset_name', type=str,
                        help='Dataset to run validation on', required=True)
    parser.add_argument('-th', '--init_threshold', type=float,
                        help='Init threshold for confidence check when adding more data', required=False, default=0.8)
    parser.add_argument('-t_dn', '--training_dataset_name', type=str,
                        help='Dataset to add as start set to train on', required=True)
    parser.add_argument("--region", type=str,
                        help="Polygon boundary of training area\n"
                             "WKT, json text or filename\n",
                        required=True)
    parser.add_argument('-m', '--model', type=str, help='The wanted neural net model to use for evaluation of dataset',
                        choices=segmentation_models.models, required=True)
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Size of minibatch', default=8)
    parser.add_argument('-f', '--features', type=int,
                        help='Number of features in first layers', default=32)
    parser.add_argument('-d', '--depth', type=int,
                        help='Depth of U', default=4)
    parser.add_argument('-a', '--activation', help='Activation function',
                        type=str, choices=segmentation_models.activations, default='relu')
    parser.add_argument('-l', '--loss', type=str,
                        choices=segmentation_models.loss_functions, default='binary_crossentropy')
    parser.add_argument('-opt', '--optimizer', type=str,
                        help='Optimizer function', default='RMSprop')
    parser.add_argument("-dc", "--dataset_config_file", type=str,
                        help="Path to data configuration file", required=False, default='config/dataset/bygg_auto_expanding.json')
    parser.add_argument('-igc', '--input_generator_config', type=str,
                        help='Path to data generator configuration', default='config/ml_input_generator/ortofoto.json')
    parser.add_argument('-test', '--in_test_mode', type=bool,
                        help='If running test mode of data teacher', default=False)
    parser.set_defaults(func=main)


def main(args):
    train_args = {
        "features": args.features,
        "depth": args.depth,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "model": args.model,
        "activation": args.activation,
        "epochs": 1 if args.in_test_mode else 100,
        "loss": args.loss,
    }
    run_data_teacher(args.data_teacher_name, args.validation_dataset_name, args.training_dataset_name, args.input_generator_config, args.dataset_config_file,
                     args.region, train_args, init_checkpoint=args.init_checkpoint, init_threshold=args.init_threshold, in_test_mode=args.in_test_mode)


def test_model_performance(new_checkpoint_name, prev_evaluation_model_checkpoint, evaluation_dataset_path, input_generator_config, model_evaluations, models_tested):
    # Check if model exist - if not the model did not improve this iteration and there is no need for prediction
    try:
        model_evaluation_res = evaluate_model(
            new_checkpoint_name, evaluation_dataset_path, input_generator_config)

        model_evaluation = model_evaluation_res["Iou_point_5"]
        evaluation_model_checkpoint = new_checkpoint_name
    except (CheckpointNotFoundException, InvalidCheckpointException):
        # Model didn't improve and therefor empty content in checkpoint-file -> error when trying to run prediction
        # no improvement, set -1 or add the last prediction value again
        model_evaluation = 0 if len(
            model_evaluations) == 0 else model_evaluations[-1]

        # Keep same eval model checkpoint for next iteration of data fetching
        evaluation_model_checkpoint = prev_evaluation_model_checkpoint

    model_evaluations.append(model_evaluation)
    models_tested.append(new_checkpoint_name)
    return model_evaluations, evaluation_model_checkpoint, models_tested


def is_model_improving(model_evaluations: list):
    min_iterations = 5
    if(len(model_evaluations) < min_iterations):
        return True
    else:
        best_model_index = model_evaluations.index(max(model_evaluations))
        total_iterations = len(model_evaluations)
        best_result_in_last_iterations = (
            total_iterations - best_model_index) < 5
        model_is_improving = best_result_in_last_iterations
        return model_is_improving


def run_data_teacher(data_teacher_name: str, validation_dataset_name: str, training_dataset_name: str, input_generator_config_path: str, dataset_config_file: str, region: str, train_args: object, init_checkpoint: str = None, init_threshold: float = None, in_test_mode: bool = False):

    evaluation_dataset_path = os.path.join(env.get_env_variable(
        'created_datasets_directory'), validation_dataset_name)

    start_trainingset_dataset_path = os.path.join(env.get_env_variable(
        'created_datasets_directory'), training_dataset_name)

    dataset_dirs = [start_trainingset_dataset_path]
    dataset_names = [training_dataset_name]

    with open(input_generator_config_path, encoding="utf8") as datagen_config:
        input_generator_config = json.load(datagen_config)

    if(init_checkpoint):
        evaluation_model_checkpoint = init_checkpoint
    else:
        # Start by training a model on the init training dataset
        evaluation_model_checkpoint = f"{data_teacher_name}_data_teacher_init"
        _train_model(evaluation_model_checkpoint, dataset_dirs,
                     dataset_names, input_generator_config, input_generator_config_path, train_args=train_args)

    models_tested = []
    model_evaluations: list = []

    # Test performance of initial model
    model_evaluations, evaluation_model_checkpoint, models_tested = test_model_performance(
        evaluation_model_checkpoint, evaluation_model_checkpoint, evaluation_dataset_path, input_generator_config, model_evaluations, models_tested)

    confidence_threshold = init_threshold
    model_is_improving = True

    max_iterations = 2 if in_test_mode else np.inf

    while (model_is_improving):
        iteration = len(model_evaluations) - 1  # first eval is from init model

        new_training_dataset_name = f"{data_teacher_name}_data_teacher_{str(iteration)}"
        new_checkpoint_name = f"{data_teacher_name}_data_teacher_{str(iteration)}"

        # Create training data where model has low confidence:
        new_train_set_path = generate_low_confidence_training_datasets(
            new_training_dataset_name, dataset_config_file, region, confidence_threshold, evaluation_model_checkpoint)

        dataset_dirs.append(new_train_set_path)
        dataset_names.append(new_training_dataset_name)

        _train_model(new_checkpoint_name, dataset_dirs, dataset_names,
                     input_generator_config, input_generator_config_path, train_args=train_args)

        model_evaluations, evaluation_model_checkpoint, models_tested = test_model_performance(
            new_checkpoint_name, evaluation_model_checkpoint, evaluation_dataset_path, input_generator_config, model_evaluations, models_tested)

        model_is_improving = is_model_improving(model_evaluations)

        # Include examples with higher confidence as dataset/model improves
        if(confidence_threshold < 0.96):
            confidence_threshold += 0.04

        print("Full list of evaluated models: ", model_evaluations)
        print("models names", models_tested)

        if(model_is_improving == False or iteration == max_iterations):
            model_is_improving = False  # To break out of while loop when max iterations reached
            best_model_index = model_evaluations.index(max(model_evaluations))
            best_model_name = models_tested[best_model_index]
            print("Model stopped improving, saving best model to azure:",
                  best_model_name)

            if in_test_mode == False:
                blobstorage.uploadModelToAzureBlobStorage(best_model_name)

            create_datateacher_metadata_file(
                best_model_name, model_evaluations, models_tested, data_teacher_name, train_args)

    return best_model_name
