
import argparse
from kartai.dataset.create_building_dataset import run_ml_predictions
from kartai.tools.predict import create_contour_result
from kartai.utils.crs_utils import get_projection_from_config_path
from kartai.utils.geometry_utils import parse_region_arg
from kartai.utils.prediction_utils import get_contour_predictions_dir, get_raster_predictions_dir
from kartai.utils.train_utils import get_existing_model_names


def add_parser(subparser):
    """Create contour vectors"""

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
    parser.add_argument('-rn', "--region_name", type=str,
                        help="Name of region that is analyzed. Used to prefix output folder in azure",
                        required=True)
    parser.add_argument("-mb", "--max_batch_size", type=int,
                        help="Max batch size for creating raster images",
                        default=200)

    parser.add_argument("-c", "--config_path", type=str,
                        help="Data configuration file", required=True)
    parser.add_argument("-si", "--start_iteration", type=int,
                        help="Start data production from this iteration. Only used if dataproduction stops, and you want to start from crahs point. NB! the resulting datasetfile will not include data before this point. ", required=False)

    '''NOT FULLY IMPLEMENTED YET parser.add_argument("-s", "--save_to", type=str, choices=['local', 'azure'], default='local',
                        help="Whether to save the resulting vector contour file to azure or locally") '''

    parser.set_defaults(func=main)


def main(args):
    """Create contour vectors"""
    geom = parse_region_arg(args.region)

    projection = get_projection_from_config_path(args.config_path)

    run_ml_predictions(args.checkpoint_name, args.region_name, projection,
                       args.config_path, geom, batch_size=args.max_batch_size, skip_data_fetching=False, start_iteration=args.start_iteration)

    raster_output_dir = get_raster_predictions_dir(
        args.region_name, args.checkpoint_name)
    contour_output_dir = get_contour_predictions_dir(
        args.region_name, args.checkpoint_name)

    print("---> Creating contour dataset from rasters")
    create_contour_result(raster_output_dir, contour_output_dir, projection)

    print("==== Contour dataset created ====")
