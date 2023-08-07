import os
import geopandas as gp
from kartai.dataset.create_building_dataset import add_confidence_values, clip_to_polygon, get_valid_geoms
from kartai.dataset.test_area_utils import get_adjusted_labels_dirs


def get_performance_count_for_detected_buildings(all_predicted_buildings, predictions_path, true_labels, new_buildings_fasit, crs, model_name, performance_output_dir, region_name):
    """Create datasets for the different category of predictions. True, false and missing"""

    print("\nCreating correctly detected buildings")
    correctly_detected_buildings = get_correctly_detected_buildings(
        true_labels, all_predicted_buildings)

    print("\nCreating correctly detected buildings that was not in FKB dataset")
    correctly_detected_new_buildings = get_correctly_detected_new_buildings(
        new_buildings_fasit, correctly_detected_buildings,  predictions_path)

    print("\nCreating falsely detected buildings")
    wrongly_detected_buildings = get_wrongly_detected_buildings(
        all_predicted_buildings, true_labels, predictions_path)

    print("\nCreating dataset of buildings not detected by the AI ")
    missing_frittliggende_buildings = get_all_missing_frittliggende_buildings(
        true_labels, all_predicted_buildings)

    print("Saving datasets")
    save_dataset(missing_frittliggende_buildings,
                 performance_output_dir,  model_name+'_missing_buildings')
    save_dataset(wrongly_detected_buildings, performance_output_dir,
                 model_name+'_false_buildings')
    save_dataset(correctly_detected_new_buildings, performance_output_dir,
                 model_name+'_true_new_buildings')
    save_dataset(all_predicted_buildings, performance_output_dir,
                 model_name+'_full_prediction_buildings')
    save_dataset(correctly_detected_buildings, performance_output_dir,
                 model_name+'_true_buildings')

    return get_dataset_count(wrongly_detected_buildings), get_dataset_count(correctly_detected_buildings),  get_dataset_count(correctly_detected_new_buildings), get_dataset_count(missing_frittliggende_buildings)


def get_dataset_count(dataset):
    if dataset.empty:
        return 0
    else:
        return len(dataset)


def save_dataset(dataset, performance_output_dir, file_name):
    os.makedirs(performance_output_dir, exist_ok=True)
    if dataset.empty:
        dataset = []
    else:
        dataset.reset_index(drop=True).to_file(
            os.path.join(performance_output_dir, file_name+'.shp'))


def get_all_missing_frittliggende_buildings(true_labels, all_predicted_buildings_dataset):
    """Get all existing buildings that was not detected"""

    missing_frittliggende_buildings = true_labels.loc[~true_labels.intersects(
        all_predicted_buildings_dataset.unary_union)]

    return missing_frittliggende_buildings


def get_wrongly_detected_buildings(all_predictions, true_labels_dataset, predictions_path):
    """Get detected buildings that does not intersect with an existing building in the manually created label dataset"""

    # Find all buildings that intersect with an existing building
    false_detected_buildings = all_predictions.loc[~all_predictions.intersects(
        true_labels_dataset.unary_union)].reset_index(drop=True)

    false_detected_buildings = add_confidence_values(
        false_detected_buildings, predictions_path)

    return false_detected_buildings


def get_correctly_detected_new_buildings(new_buildings_fasit, true_predicted_buildings, predictions_path):
    """Get buildings that are correctly identified, but that are not in FKB"""

    # Find detected buildings that overlap actual new buildings
    true_new_buildings = true_predicted_buildings.loc[true_predicted_buildings.intersects(
        new_buildings_fasit.unary_union)]

    # Dissolve polygons that belong to the same building in labels, to avoid several counts for the same detection
    true_new_buildings = true_new_buildings.dissolve(by='label_id')

    # Adding confidence values from AI analysis
    true_new_buildings = add_confidence_values(
        true_new_buildings, predictions_path)
    return true_new_buildings


def get_true_labels(region_name, region, crs):
    true_labels_dataset = get_adjusted_labels(
        region_name, crs)

    # Add a unique ID to all rows in the dataset
    true_labels_dataset['label_id'] = range(len(true_labels_dataset))
    true_labels_dataset.set_index("label_id")
    if true_labels_dataset.geometry.area.hasnans:
        print("Some columns are missing area value - add them before using the adjusted dataset. Run a dissolve in qgis!")

    # Clip to the region polygon:
    true_labels_dataset = clip_to_polygon(true_labels_dataset, region, crs)

    return true_labels_dataset


def get_adjusted_labels(region_name, crs):
    """ Path to manually adjusted labels where every building is correctly annotated """
    adjusted_label_dirs = get_adjusted_labels_dirs(region_name)
    adjusted_labels_gdf = get_geo_data_frame(adjusted_label_dirs, crs)
    adjusted_labels_gdf["geometry"] = get_valid_geoms(adjusted_labels_gdf)
    return adjusted_labels_gdf


def get_geo_data_frame(file_dirs, crs, driver="shapefile"):
    gdf = gp.GeoDataFrame()

    for file in file_dirs:
        gdf = gdf.append(
            gp.read_file(file, driver=driver), ignore_index=True)

    gdf.set_crs(crs, inplace=True)
    return gdf


def get_correctly_detected_buildings(true_labels, all_predicted_buildings_dataset):
    """Get all predictions that are actual buildings, according to the adjusted manually created label dataset"""

    # Find all buildings that intersect with an existing building
    true_predictions = all_predicted_buildings_dataset.loc[all_predicted_buildings_dataset.intersects(
        true_labels.unary_union)].reset_index(drop=True)

    # Clip the predictions to the true_labels, in order to keep the label_id column
    true_predictions = gp.overlay(
        true_predictions, true_labels, how='intersection')
    # Dissolve geometry that belongs to the same building, meaning they have the same label_id
    true_predictions = true_predictions.dissolve(by="label_id")

    return true_predictions


def get_new_buildings_fasit(true_labels, fkb_labels):
    """Get all buildings in manually adjusted labels dataset, that is not in FKB data"""
    true_new_frittliggende_buildings = true_labels.loc[~true_labels.intersects(
        fkb_labels.unary_union)]
    return true_new_frittliggende_buildings
