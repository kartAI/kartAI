import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from azure import blobstorage
from tensorflow import keras

import env
from kartai.dataset import dataGenerator
from kartai.utils.dataset_utils import validate_model_data_input
from kartai.metrics.meanIoU import (Iou_point_5, Iou_point_6, Iou_point_7,
                                    Iou_point_8, Iou_point_9, IoU, IoU_fz)
from kartai.models import segmentation_models
from kartai.utils.train_utils import check_for_existing_model, get_dataset_dirs, get_existing_model_names

''' Start training with a selected model and dataset
    For now only running u-net, but in the future we should instead pass input to define the model and dataset once running the script
'''


def train_model(created_datasets_dirs, input_generator_config, checkpoint_name, model_name, features, depth, activation_name, batch_size, epochs, optimizer, loss_function, checkpoint_to_finetune_from=False):
    # Build model
    num_classes = 1

    input1_size, input2_size = get_input_dimensions(
        input_generator_config, model_name)

    if activation_name and activation_name in segmentation_models.activations:
        activation = segmentation_models.activations[activation_name]
    else:
        print(f'Unknown activation "{activation_name}"', file=sys.stderr)
        sys.exit(1)

    if model_name and model_name in segmentation_models.models:
        model_fn = segmentation_models.models[model_name]
    else:
        print(f'Unknown model "{model_name}"', file=sys.stderr)
        sys.exit(1)

    model = model_fn(input1_size, num_classes, activation,
                     features, depth, input2_size) if input2_size else model_fn(input1_size, num_classes, activation,
                                                                                features, depth)
    model.summary()

    # Define optimizer:
    opt = getOptimizer(
        optimizer, is_finetuning=checkpoint_to_finetune_from != False)

    checkpoint_path = env.get_env_variable('trained_models_directory')
    log_path = 'tensorflow_log'

    # Load weights from previously trained model
    if(checkpoint_to_finetune_from):
        print(
            f'\n --- Loading weights from model {checkpoint_to_finetune_from} to finetune from --- \n')
        model.load_weights(os.path.join(
            checkpoint_path, checkpoint_to_finetune_from+'.h5'))

    # Define loss-function:
    loss = getLoss(loss_function)

    # Configure the model for training:
    model.compile(optimizer=opt, loss=loss,
                  metrics=[keras.metrics.BinaryAccuracy(), IoU, IoU_fz, Iou_point_5, Iou_point_6, Iou_point_7, Iou_point_8, Iou_point_9])

    os.makedirs(log_path, exist_ok=True)

    checkpoint_dir = os.path.join(checkpoint_path, checkpoint_name+".h5")
    print('\n----SAVE TO:', checkpoint_dir + '\n')
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_iou_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        monitor="val_IoU",
        mode='max',
        verbose=1,
        save_best_only=True)

    earlystop_iou_cb = keras.callbacks.EarlyStopping(
        monitor="val_IoU",
        min_delta=0,
        patience=15,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=False,
    )

    reduceLR_loss_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_IoU", factor=0.2, patience=6, verbose=0, min_delta=0)

    if(epochs == None):
        epochs = 100  # Max number of epochs

    callbacks = [
        checkpoint_iou_cb,
        reduceLR_loss_cb,
        earlystop_iou_cb
    ]

    # Train the model, doing validation at the end of each epoch.

    trainGenerator = dataGenerator.DataGenerator(input_generator_config, num_classes,
                                                 'train', created_datasets_dirs, batch_size)

    validationGenerator = dataGenerator.DataGenerator(input_generator_config, num_classes,
                                                      'validation', created_datasets_dirs, batch_size)

    train_history = model.fit(trainGenerator, epochs=epochs, shuffle=False,
                              validation_data=validationGenerator, callbacks=callbacks)

    return train_history


def get_input_dimensions(input_generator_config, model_name):
    input_stack = input_generator_config["model_input_stack"]
    input_tuple = input_generator_config["model_input_tuple"]

    validate_model_data_input(input_generator_config,
                              model_name, segmentation_models)

    model_input = input_stack if len(input_stack) > 0 else input_tuple
    if(len(input_tuple) > 0):
        input1 = (model_input[0]["dimensions"][0], model_input[0]
                  ["dimensions"][1], model_input[0]["dimensions"][2])
        input2 = (model_input[1]["dimensions"][0], model_input[1]
                  ["dimensions"][1], model_input[1]["dimensions"][2])
    else:
        # stack
        totalNumOfChannels = 0
        for inp in model_input:
            totalNumOfChannels += inp["dimensions"][2]

        input1 = (model_input[0]["dimensions"][0], model_input[0]["dimensions"]
                  [1], totalNumOfChannels)
        input2 = None

    return input1, input2


def create_metadata_file(dataset_dirs, datasets_to_train_on, datagenerator_config_name, checkpoint_name, model, features,
                         depth, activation, batch_size, epochs, train_history, optimizer, loss_function, checkpoint_to_finetune_from=None):

    ct = datetime.datetime.now()
    dataset_ids = []
    for dataset_dir in dataset_dirs:
        with open(os.path.join(dataset_dir, 'meta.json'), 'r') as f:
            dataset_meta_data = json.load(f)
            dataset_ids.append(dataset_meta_data['id'])

    for i in range(len(train_history.history["lr"])):
        train_history.history["lr"][i] = float(train_history.history["lr"][i])

    meta = {
        "checkpoint_name": checkpoint_name,
        "date_time": str(ct),
        "model": model,
        "datagenerator_config": datagenerator_config_name,
        "features": features,
        "depth": depth,
        "activation": activation,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "loss": loss_function,
        "finetuned_from": checkpoint_to_finetune_from,
        "training_dataset_id": ', '.join(str(x) for x in dataset_ids),
        "training_dataset_name": ', '.join(str(x) for x in datasets_to_train_on),
        "training_results": train_history.history,
        "epochs": epochs,
        "early_stopping": {
            "monitor": "val_Iou_point_5",
            "min_delta": 0,
            "patience": 15,
            "verbose": 1,
            "mode": "max",
            "baseline": "None",
            "restore_best_weights": "False",
        },
        "learning_rate_decay": {
            "monitor": 'val_Iou_point_5',
            "factor": 0.2,
            "patience": 6,
            "verbose": 0,
            "min_delta": 0
        }
    }
    ident = 2
    file = os.path.join(env.get_env_variable(
        'trained_models_directory'), checkpoint_name+'.meta.json')

    with open(file, 'w') as outfile:
        json.dump(meta, outfile, indent=ident)

    print(json.dumps(meta,  indent=ident))


def getOptimizer(optimizer, is_finetuning):

    learning_rate = 0.0005 if is_finetuning else 0.001

    print('learning_rate', learning_rate)

    optimizers = {
        "RMSprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
        "Adam": keras.optimizers.Adam(learning_rate=learning_rate),
        "Adadelta": keras.optimizers.Adadelta(learning_rate=learning_rate),
        'Adagrad': keras.optimizers.Adagrad(learning_rate=learning_rate),
        'Adamax': keras.optimizers.Adamax(learning_rate=learning_rate),
        'Nadam': keras.optimizers.Nadam(learning_rate=learning_rate),
        'Ftrl': keras.optimizers.Ftrl(learning_rate=learning_rate),
    }

    if(optimizer not in optimizers.keys()):
        print(
            f'Optimizer {optimizer} is not available. Choose from: ' + optimizers.keys())

    return optimizers[optimizer]


def getLoss(loss_function):
    if loss_function == 'binary_crossentropy':
        loss_function = 'binary_crossentropy'
    elif loss_function == 'focal_loss':
        from kartai.losses.binary_focal_loss import binary_focal_loss
        loss_function = binary_focal_loss()

    return loss_function


def add_parser(subparser):
    parser = subparser.add_parser(
        "train",
        help="train models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    existing_trained_model_names = get_existing_model_names()

    parser.add_argument('-dn', '--dataset_name', action='append',
                        help='Name or list for training datasets to train with', required=True)
    parser.add_argument('-cn', '--checkpoint_name', type=str,
                        help='Name for the resulting checkpoint file', required=True)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--no-save_model',
                        dest='save_model', action='store_false')
    parser.set_defaults(save_model=True)
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Size of minibatch', default=8)
    parser.add_argument('-e', '--epochs', type=int,
                        help='Number of epochs', default=100)
    parser.add_argument('-m', '--model', type=str,
                        help='The wanted neural net model', choices=segmentation_models.models, required=True)
    parser.add_argument('-f', '--features', type=int,
                        help='Number of features in first layers', default=32)
    parser.add_argument('-d', '--depth', type=int,
                        help='Depth of U', default=4)
    parser.add_argument('-a', '--activation', type=str,
                        help='Activation function', choices=segmentation_models.activations, default='relu')
    parser.add_argument('-l', '--loss', type=str,
                        choices=segmentation_models.loss_functions, default='binary_crossentropy')
    parser.add_argument('-ft', '--checkpoint_to_finetune', type=str,
                        help='Name of checkpoint to finetune the model with', choices=existing_trained_model_names, default=False)
    parser.add_argument('-opt', '--optimizer', type=str,
                        help='Optimizer function', default='RMSprop')
    parser.add_argument('-c', '--config', type=str,
                        help='Path to data generator configuration', default='config/ml_input_generator/ortofoto.json')
    parser.set_defaults(func=main)


def main(args):

    train_args = {
        "features": args.features,
        "depth": args.depth,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "model": args.model,
        "loss": args.loss,
        "activation": args.activation,
        "epochs": args.epochs
    }

    train(args.checkpoint_name, args.dataset_name,
          args.config, args.save_model, train_args, args.checkpoint_to_finetune)


def train(checkpoint_name, dataset_name, input_generator_config_path, save_model, train_args, checkpoint_to_finetune):

    check_for_existing_model(checkpoint_name)

    created_datasets_dirs = get_dataset_dirs(dataset_name)

    with open(input_generator_config_path, encoding="utf8") as config:
        input_generator_config = json.load(config)

    train_history = train_model(created_datasets_dirs, input_generator_config, checkpoint_name, train_args['model'],
                                train_args['features'], train_args['depth'], train_args['activation'], train_args['batch_size'], train_args['epochs'], train_args['optimizer'], train_args['loss'], checkpoint_to_finetune_from=checkpoint_to_finetune)

    create_metadata_file(created_datasets_dirs, dataset_name, Path(input_generator_config_path).stem, checkpoint_name, train_args['model'], train_args['features'], train_args['depth'], train_args[
                         'activation'], train_args['batch_size'], train_args['epochs'], train_history, train_args['optimizer'], train_args['loss'], checkpoint_to_finetune_from=checkpoint_to_finetune)

    if save_model:
        blobstorage.uploadModelToAzureBlobStorage(checkpoint_name)
