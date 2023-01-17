from azure import blobstorage
import argparse


def add_parser(subparser):
    parser = subparser.add_parser(
        "upload_model",
        help="upload model to azuer (h5 file and metadata)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-m', '--model_name', type=str,
                        help='Name of the model to upload', required=True)

    parser.set_defaults(func=main)


def main(args):
    blobstorage.uploadModelToAzureBlobStorage(args.model_name)
