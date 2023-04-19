import os
import env
import argparse
from kartai.dataset.create_building_dataset import create_building_dataset
from kartai.dataset.create_contour_dataset import create_building_contour_dataset
from kartai.utils.geometry_utils import parse_region_arg
from kartai.utils.train_utils import get_existing_model_names


def add_parser(subparser):
    parser = subparser.add_parser(
        "create_predicted_buildings_contour",
        help="Create vector contour from ML predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    existing_trained_model_names = get_existing_model_names()

    parser.add_argument('-cn', '--checkpoint_name', type=str, choices=existing_trained_model_names,
                        help='Name for checkpoint file to use for prediction', required=True)

    parser.add_argument('-n', '--output_name', type=str,
                        help='Name for output dataset', required=True)
    parser.add_argument("--region", type=str,
                        help="Polygon boundary of training area\n"
                             "WKT, json text or filename\n"
                             "alternative to bounding box",
                        required=True)
    parser.add_argument('-an', "--area-name", type=str,
                        help="Name of area that is analyzed. Used to prefix output folder in azure",
                        required=True)
    parser.add_argument("-mb", "--max_mosaic_batch_size", type=int,
                        help="Max batch size for creating mosaic of the predictions",
                        default=200)

    parser.add_argument("-c", "--config_path", type=str,
                        help="Data configuration file", required=True)

    parser.add_argument("-p", "--skip_to_postprocess", type=str, required=False, default='false',
                        help="Whether to skip directly to postprocessing, and not look for needed downloaded data. Typically used if you have already run production of dataset for same area, but with different model")

    parser.add_argument("-s", "--save_to", type=str, choices=['local', 'azure'], default='azure',
                        help="Whether to save the resulting vector contour file to azure or locally")

    parser.set_defaults(func=main)


def main(args):
    output_dir = os.path.join(env.get_env_variable(
        "prediction_results_directory"), args.output_name)

    geom = parse_region_arg(args.region)

    skip_to_postprocess = False if args.skip_to_postprocess == 'false' or args.skip_to_postprocess == None else True

    print('skip_to_postprocess', skip_to_postprocess)

    create_building_contour_dataset(geom, args.checkpoint_name, args.area_name,
                                    args.config_path, skip_to_postprocess=skip_to_postprocess, output_dir=output_dir, max_mosaic_batch_size=args.max_mosaic_batch_size, save_to=args.save_to)
