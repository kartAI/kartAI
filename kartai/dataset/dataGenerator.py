import sys

import numpy as np
from tensorflow import keras as K
from PIL import Image
import json
import os
import tensorflow as tf
from kartai.utils.dataset_utils import get_X_tuple, get_ground_truth, get_X_stack
from kartai.datamodels_and_services.ImageSourceServices import Tile

class DataGenerator(K.utils.Sequence):
    def __init__(self, datagenerator_config, num_classes, dataset_type, created_datasets_dirs, batch_size, shuffle=True):

        filename = 'train_set.json' if dataset_type == 'train' else 'valid_set.json'

        input_list = []

        for created_dataset in created_datasets_dirs:
            with open(os.path.join(created_dataset, filename), 'r') as input_data:
                input_list_dataset_json = json.load(input_data)
                input_list_dataset = Tile.tileset_from_json(input_list_dataset_json)
                input_list = input_list + input_list_dataset

        self.datagenerator_config = datagenerator_config
        self.num_classes = num_classes
        self.input_list = input_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.input_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # number of indexes in one epoch
        return int(np.floor(len(self.input_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        tileset_for_batch = [self.input_list[k] for k in indexes]

        if(len(self.datagenerator_config["model_input_stack"])):
            return self._get_stack_item(
                tileset_for_batch, self.datagenerator_config["model_input_stack"], self.datagenerator_config["ground_truth"])

        elif(len(self.datagenerator_config["model_input_tuple"])):
            return self._get_tuple_item(
                tileset_for_batch, self.datagenerator_config["model_input_tuple"], self.datagenerator_config["ground_truth"])

        else:
            print(
                'Missing either model_input_stack or model_input_tuple in datagenerator config')
            sys.exit(1)

    def _get_tuple_item(self, tileset_for_batch, tuple_input, ground_truth):
        X_tuple = get_X_tuple(
            self.batch_size, tileset_for_batch, tuple_input)
        Y_batch = get_ground_truth(
            self.batch_size, tileset_for_batch, ground_truth)
        return X_tuple, Y_batch

    def _get_stack_item(self, tileset_for_batch, input_stack, ground_truth):
        X_batch = get_X_stack(
            self.batch_size, tileset_for_batch, input_stack)
        Y_batch = get_ground_truth(
            self.batch_size, tileset_for_batch, ground_truth)

        return X_batch, Y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
