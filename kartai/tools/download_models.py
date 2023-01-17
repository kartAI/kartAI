from azure import blobstorage
import os
import env
import argparse


def add_parser(subparser):
    parser = subparser.add_parser(
        "download_models",
        help="Download all models from azure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=main)


def main(args):

    output_directory = env.get_env_variable(
        'trained_models_directory')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    blobstorage.downloadTrainedModels()
