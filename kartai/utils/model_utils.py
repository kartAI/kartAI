import os
from tensorflow import keras
from kartai.utils.confidence import Confidence
from kartai.utils.dataset_utils import get_ground_truth, get_X_stack, get_X_tuple
from kartai.metrics.meanIoU import (IoU, IoU_fz, Iou_point_5, Iou_point_6,
                                    Iou_point_7, Iou_point_8, Iou_point_9)
from kartai.tools.train import get_loss
import numpy as np
import env


def predict_image(image_path, lidar_path, label_path, datagenerator_config, checkpoint_to_predict_with):
    model = load_checkpoint_model(checkpoint_to_predict_with)

    input_images = _get_input_images(
        [{"image": image_path, "label": label_path, "lidar": lidar_path}], datagenerator_config)
    # Predict test data from model
    print('input images', input_images)
    predictions = model.predict(input_images, 6)

    return predictions


def evaluate_image(image_path, label_path, datagenerator_config, checkpoint_to_predict_with):
    model = load_checkpoint_model(checkpoint_to_predict_with)

    input_labels = get_ground_truth(
        1, [image_path], datagenerator_config["ground_truth"])

    input_images = _get_input_images(
        [{"image": image_path, "label": label_path[0]}], datagenerator_config)

    # Evaluate model on test data
    results = model.evaluate(
        input_images, input_labels, 6, return_dict=True)

    return results


def model_is_confident(data_generator_config, ortofoto_path: str, lidar_path: str, label_path: str, eval_model_checkpoint, confidence_threshold: float):

    if (confidence_threshold == None):
        raise Exception("Missing threshold for confidence")

    prediction = predict_image(
        ortofoto_path, lidar_path, label_path, data_generator_config, eval_model_checkpoint)
    confidence = np.mean(np.absolute(prediction - 0.5) * 2)

    print('confidence_value', confidence)
    return confidence > confidence_threshold


def load_checkpoint_model(checkpoint_name: str) -> keras.Sequential:
    checkpoint_path = env.get_env_variable(
        'trained_models_directory')
    model_fn = os.path.join(checkpoint_path, checkpoint_name + '.h5')
    dependencies = {
        'BinaryFocalLoss': get_loss('focal_loss'),
        'Iou_point_5': Iou_point_5,
        'Iou_point_6': Iou_point_6,
        'Iou_point_7': Iou_point_7,
        'Iou_point_8': Iou_point_8,
        'Iou_point_9': Iou_point_9,
        'IoU': IoU,
        'IoU_fz': IoU_fz,
        "Confidence": Confidence()
    }

    model = keras.models.load_model(model_fn, custom_objects=dependencies)

    return model


def checkpoint_exist(checkpoint_name):
    checkpoint_path = env.get_env_variable(
        'trained_models_directory')
    model_path = os.path.join(checkpoint_path, checkpoint_name + '.h5')
    file_exist = os.path.exists(model_path)
    return file_exist

# Input paths are a list of objects with fields {"image": "..", "label": "..", "lidar"}


def _get_input_images(input_paths: list[dict], datagenerator_config: dict):
    batch_size = len(input_paths)
    has_tuple_input = len(datagenerator_config["model_input_tuple"]) > 0

    if (has_tuple_input):
        input_images = get_X_tuple(
            batch_size, input_paths, datagenerator_config["model_input_tuple"])
    else:
        input_images = get_X_stack(
            batch_size, input_paths, datagenerator_config["model_input_stack"])

    return input_images
