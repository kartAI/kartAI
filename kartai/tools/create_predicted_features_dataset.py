import argparse
import time
from kartai.dataset.create_polygon_dataset import produce_vector_dataset, run_ml_predictions
from kartai.utils.config_utils import read_config
from kartai.utils.crs_utils import get_projection_from_config_path
from kartai.utils.geometry_utils import parse_region_arg
from kartai.utils.prediction_utils import get_raster_predictions_dir, get_vector_predictions_dir
from kartai.utils.train_utils import get_existing_model_names


def add_parser(subparser):
    parser = subparser.add_parser(
        "create_predicted_features_dataset",
        help="Create dataset with features from ML prediction",
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
    parser.add_argument("-mb", "--max_mosaic_batch_size", type=int,
                        help="Max batch size for creating mosaic of the predictions",
                        default=200)
    parser.add_argument("-c", "--config_path", type=str,
                        help="Data configuration file", default="config/dataset/bygg-no-rules.json")
    parser.add_argument("-mp", "--num_load_processes", type=int, default=1,
                        help="Number of parallell processes to run when downloading training data")
    parser.add_argument("-s", "--save_to", type=str, choices=['local', 'azure'], default='azure',
                        help="Whether to save the resulting vector data to azure or locally")

    parser.set_defaults(func=main)


def main(args):

    geom = parse_region_arg(args.region)

    config = read_config(args.config_path)

    projection = get_projection_from_config_path(args.config_path)

    """ run_ml_predictions(args.checkpoint_name, args.region_name, projection,
                       config=config, geom=geom, num_processes=args.num_load_processes) """

    time.sleep(2)  # Wait for complete saving to disk

    print('Starting postprocess')

    vector_output_dir = get_vector_predictions_dir(
        args.region_name, args.checkpoint_name)
    raster_predictions_path = get_raster_predictions_dir(
        args.region_name, args.checkpoint_name)

    produce_vector_dataset(
        vector_output_dir, raster_predictions_path, config, args.max_mosaic_batch_size, f"{args.region_name}_{args.checkpoint_name}", save_to="local")
