
import argparse
from kartai.dataset.create_building_dataset import run_ml_predictions
from kartai.tools.predict import create_contour_result
from kartai.utils.config_utils import read_config
from kartai.utils.crs_utils import get_projection_from_config_path
from kartai.utils.geometry_utils import parse_region_arg
from kartai.utils.prediction_utils import get_contour_predictions_dir, get_raster_predictions_dir
from kartai.utils.train_utils import get_existing_model_names


def add_parser(subparser):
    """Create vectordata by createing contours from the probability/confidence lever from the AI"""

    parser = subparser.add_parser(
        "create_predicted_buildings_contour",
        help="Create vector contour from ML predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    existing_trained_model_names = get_existing_model_names()

    parser.add_argument('-cn', '--input_model_name', type=str, choices=existing_trained_model_names,
                        help='Name for checkpoint file to use for prediction', required=True)
    parser.add_argument('-im_sub', '--input_model_subfolder', type=str,
                        help='If model is saved with new directory structure, specify the name of the subfolder containing the checkpoint')
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
    parser.add_argument("-mp", "--num_load_processes", type=int, default=1,
                        help="Number of parallell processes to run when downloading training data")
    parser.add_argument("-c", "--config_path", type=str,
                        help="Data configuration file", default="config/dataset/bygg-no-rules.json")
    parser.add_argument("-l", "--contour_levels", type=str, default="0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1",
                        help="The confidence levels to create contours for. String with comma seperated float number", required=False)

    parser.set_defaults(func=main)


def main(args):
    """Create contour vectors"""
    geom = parse_region_arg(args.region)

    projection = get_projection_from_config_path(args.config_path)

    config = read_config(args.config_path)

    run_ml_predictions(args.input_model_name, args.region_name, projection, args.input_model_subfolder,
                       config=config, geom=geom, batch_size=args.max_batch_size, skip_data_fetching=False,
                       save_to="local", num_processes=args.num_load_processes)

    raster_output_dir = get_raster_predictions_dir(
        args.region_name, args.input_model_name)
    contour_output_dir = get_contour_predictions_dir(
        args.region_name, args.input_model_name)

    print("---> Creating contour dataset from rasters")
    contour_levels = str.split(args.contour_levels, ",")
    contour_levels = [float(level) for level in contour_levels]
    create_contour_result(
        raster_output_dir, contour_output_dir, projection, contour_levels)

    print("==== Contour dataset created ====")
