

import os
from azure import blobstorage
import argparse
import env


def add_parser(subparser):
    parser = subparser.add_parser(
        "upload_files_in_dir",
        help="upload all files in a given directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    building_dataset_contianer = env.get_env_variable(
        "results_datasets_container_name")

    parser.add_argument('-p', '--path', type=str,
                        help='Path to directory', required=True)
    parser.add_argument('-c', '--container_name', type=str,
                        help='Name of the azure container to upload data to', required=False, default=building_dataset_contianer)

    parser.set_defaults(func=main)


def main(args):
    # building-predictions-contour
    for filename in os.listdir(args.path):
        with open(os.path.join(args.path, filename), 'r') as file:
            blobstorage.upload_data_to_azure(
                file, filename, args.container_name)
