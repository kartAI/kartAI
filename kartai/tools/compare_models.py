
import argparse
import os

from kartai.models import segmentation_models


def test_all_models_with_stacked_data(dataset, name_suffix=None):
    batch = 4
    activations = ["mish", "swish", "relu"]
    depths = [3, 4, 5, 6]

    models = segmentation_models.models
    # Remove twin model which is not compatible with the current data
    del models["unet-twin"]

    for model_name in models:
        for activation in activations:
            for depth in depths:
                if(depth > 4):
                    features = 8
                else:
                    features = 16

                name = f"{model_name}_{dataset}_d{depth}_{activation}{name_suffix}"

                os.system(
                    f"./kai train -cn {name} -m {model_name} -dn {dataset} -a {activation} -d {depth} -bs {batch} -f {features}"
                )


def test_several_datasets(dataset, name_suffix=None):
    batch = 4
    activations = ["swish", "relu"]
    depths = [4]

    dataset1 = "full_analysis_unet_v1_data_teacher_1"
    dataset2 = "full_analysis_unet_v1_data_teacher_0"
    dataset3 = "building_area"
    dataset_name = "building_area_and_data_teacher_datasets"

    models = ["CSP_cross_SPP", "unet"]

    for model_name in models:
        for activation in activations:
            for depth in depths:
                if(depth > 4):
                    features = 8
                else:
                    features = 16

                name = f"{model_name}_{dataset_name}_d{depth}_{activation}{name_suffix}"

                os.system(
                    f"./kai train -cn {name} -m {model_name} -dn {dataset1} -dn {dataset2} -dn {dataset3} -a {activation} -d {depth} -bs {batch} -f {features}"
                )


def add_parser(subparser):
    parser = subparser.add_parser(
        "compare_models",
        help="Run a set of models for testing different architectures / hyperparameters / dataset and so on",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=main)


def main(args):
    dataset = "large_building_area"
    test_all_models_with_stacked_data(dataset, "")
