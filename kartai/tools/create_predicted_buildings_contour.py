import os
import env
import argparse
from kartai.dataset.create_building_dataset import run_ml_predictions
from kartai.tools.predict import create_contour_result
from kartai.utils.geometry_utils import parse_region_arg
from kartai.utils.prediction_utils import get_contour_predictions_dir, get_raster_predictions_dir
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
                        default=40)

    parser.add_argument("-c", "--config_path", type=str,
                        help="Data configuration file", required=True)

    parser.add_argument("-df", "--skip_data_fetching", type=str, required=False, default='false',
                        help="Whether to skip directly to running ML prediction, and not look for needed downloaded data. Typically used if you have already run production of dataset for same area, but with different model")

    parser.add_argument("-s", "--save_to", type=str, choices=['local', 'azure'], default='local',
                        help="Whether to save the resulting vector contour file to azure or locally")

    parser.set_defaults(func=main)


def main(args):
    ''' 
        Det må være samme datastruktur uavhengig av hvilken postprosessering man ønkser å benytte.

        Kan strukturen være:

        - results 
          - area_name
            -checkpoint_name
              -rasters
              -contour
              -vector

    '''

    geom = parse_region_arg(args.region)

    skip_data_fetching = False if args.skip_data_fetching == 'false' or args.skip_data_fetching == None else True

    skip_to_postprocess = False  # TODO: get from args

    print('skip_data_fetching', skip_data_fetching)

    if skip_to_postprocess == False:
        projection = run_ml_predictions(args.checkpoint_name, args.area_name,
                                        args.config_path, geom, batch_size=args.max_mosaic_batch_size, skip_data_fetching=skip_data_fetching, save_to=args.save_to)

    raster_output_dir = get_raster_predictions_dir(
        args.area_name, args.checkpoint_name)
    contour_output_dir = get_contour_predictions_dir(
        args.area_name, args.checkpoint_name)

    create_contour_result(raster_output_dir, contour_output_dir, projection)

    print("==== Contour dataset created ====")
