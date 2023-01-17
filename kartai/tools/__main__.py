import argparse
import sys
import os


def add_parsers():
    parser = argparse.ArgumentParser(prog="./kai")
    subparser = parser.add_subparsers(title="kartai tools", metavar="")
    tool = sys.argv[1]

    # Add your tool's entry point below.
    if tool == "compare_models":
        from kartai.tools import compare_models
        compare_models.add_parser(subparser)
    elif tool == "create_training_data":
        from kartai.tools import create_training_data
        create_training_data.add_parser(subparser)
    elif tool == "predict":
        from kartai.tools import predict
        predict.add_parser(subparser)
    elif tool == "train":
        from kartai.tools import train
        train.add_parser(subparser)
    elif tool == "results":
        from kartai.tools import results
        results.add_parser(subparser)
    elif tool == "download_models":
        from kartai.tools import download_models
        download_models.add_parser(subparser)
    elif tool == "create_predicted_buildings_dataset":
        from kartai.tools import create_predicted_buildings_dataset
        create_predicted_buildings_dataset.add_parser(subparser)
    elif tool == "data_teacher":
        from kartai.tools import data_teacher
        data_teacher.add_parser(subparser)
    elif tool == "upload_model":
        from kartai.tools import upload_model
        upload_model.add_parser(subparser)
    elif tool == "full_analysis":
        from kartai.tools import full_analysis
        full_analysis.add_parser(subparser)

    # We return the parsed arguments, but the sub-command parsers
    # are responsible for adding a function hook to their command.

    subparser.required = True

    return parser.parse_args()


def main():
    """main entrypoint for kartai tools"""
    args = add_parsers()

    validate_existing_paths(args)
    # Check for valid paths in args

    args.func(args)


def validate_existing_paths(args):

    path_params = ["config", "config_path", "input_generator_config_path",
                   "dataset_config_file", "region"]

    args_obj = vars(args)

    for param in path_params:
        if(param in args_obj and not os.path.exists(args_obj[param])):
            raise Exception(f"{param} does not exist in path", args_obj[param])


if __name__ == "__main__":
    main()
