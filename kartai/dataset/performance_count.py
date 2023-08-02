import os
import uuid
import geopandas as gp
from kartai.dataset.Iou_calculations import get_geo_data_frame
from kartai.dataset.create_building_dataset import get_valid_geoms, perform_last_adjustments
from kartai.dataset.test_area_utils import get_adjusted_labels_dir, get_test_region_avgrensning_dir


def get_performance_count_for_detected_buildings(all_predicted_buildings_dataset, predictions_path, true_labels, new_buildings_fasit, CRS_prosjektomrade, model_name, performance_output_dir, region_name):

    correctly_detected_buildings = get_correctly_detected_buildings(
        true_labels, all_predicted_buildings_dataset)

    correctly_detected_new_buildings = get_correctly_detected_new_buildings(
        new_buildings_fasit, all_predicted_buildings_dataset,  predictions_path, CRS_prosjektomrade)

    wrongly_detected_buildings = get_wrongly_detected_frittliggende_bygg_dataset(
        all_predicted_buildings_dataset, true_labels, predictions_path, CRS_prosjektomrade)

    missing_frittliggende_buildings = get_all_missing_frittliggende_buildings(
        true_labels, all_predicted_buildings_dataset)

    # Cut all datasets to test-region - the fetched prediction tiles cover area outside as well
    area_to_predict = get_test_region_avgrensning_dir(region_name) #TODO: use region instead
    
    area_gdf = gp.read_file(area_to_predict)
    wrongly_detected_buildings = gp.clip(
        wrongly_detected_buildings, area_gdf)
    all_predicted_buildings_dataset = gp.clip(
        all_predicted_buildings_dataset, area_gdf)

    save_dataset(missing_frittliggende_buildings,
                 performance_output_dir,  model_name+'_missing_buildings')
    save_dataset(wrongly_detected_buildings, performance_output_dir,
                 model_name+'_false_buildings')
    save_dataset(correctly_detected_new_buildings, performance_output_dir,
                 model_name+'_true_new_buildings')
    save_dataset(all_predicted_buildings_dataset, performance_output_dir,
                 model_name+'_full_prediction_buildings')
    save_dataset(correctly_detected_buildings, performance_output_dir,
                 model_name+'_true_buildings')

    return get_dataset_count(wrongly_detected_buildings), get_dataset_count(correctly_detected_buildings),  get_dataset_count(correctly_detected_new_buildings),get_dataset_count(missing_frittliggende_buildings)


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
    
    missing_frittliggende_buildings = true_labels.loc[~true_labels.intersects(all_predicted_buildings_dataset.unary_union)].reset_index(drop=True)

    return missing_frittliggende_buildings


def get_wrongly_detected_frittliggende_bygg_dataset(all_predictions, true_labels_dataset, predictions_path, crs):
    """Get detected buildings that does not intersect with an existing building in the manually created label dataset"""
    
    #Find all buildings that intersect with an existing building
    false_detected_buildings = all_predictions.loc[~all_predictions.intersects(true_labels_dataset.unary_union)].reset_index(drop=True)

    # Dissolve polygons that belong to the same building in labels, to avoid several counts for the same detection
    false_detected_buildings = false_detected_buildings.dissolve(by='label_id')
    false_detected_buildings=perform_last_adjustments(false_detected_buildings, predictions_path, crs)

    return false_detected_buildings


def get_correctly_detected_new_buildings(new_buildings_fasit, all_predicted_buildings_dataset, predictions_path, crs):
    """Get buildings that are correctly identified, but that are not in FKB"""
    
    # Find detected buildings that overlap actual new buildings
    true_detected_buildings = all_predicted_buildings_dataset.loc[all_predicted_buildings_dataset.intersects(new_buildings_fasit.unary_union)]

    # Dissolve polygons that belong to the same building in labels, to avoid several counts for the same detection
    true_detected_buildings = true_detected_buildings.dissolve(by='label_id')
    
    # Adding prob value, removing small buildings ++
    true_detected_buildings = perform_last_adjustments(
        true_detected_buildings, predictions_path, crs)
    return true_detected_buildings


def get_true_labels(region_name, CRS_prosjektomrade):
    true_labels_dataset = get_adjusted_labels(
        region_name, CRS_prosjektomrade)
    
    #Add a unique ID to all rows in the dataset
    true_labels_dataset['label_id'] = true_labels_dataset.apply(lambda _: uuid.uuid4(), axis=1)
    true_labels_dataset.set_index("label_id")

    if true_labels_dataset.geometry.area.hasnans:
        print("Some columns are missing area value - add them before using the adjusted dataset. Run a dissolve in qgis!")

    return true_labels_dataset


def get_adjusted_labels(region_name, CRS_prosjektomrade):
    """ Path to manually adjusted labels where every building is correctly annotated """
    adjusted_label_dir = get_adjusted_labels_dir(region_name)
    adjusted_labels_gdf = get_geo_data_frame(adjusted_label_dir, CRS_prosjektomrade)
    adjusted_labels_gdf["geometry"] = get_valid_geoms(adjusted_labels_gdf)
    return adjusted_labels_gdf


def get_correctly_detected_buildings(true_labels_dataset, all_predicted_buildings_dataset):
    """Get all predictions that are actual buildings, according to the adjusted manually created label dataset"""
    
    #Find all buildings that intersect with an existing building
    true_new_frittliggende_buildings = all_predicted_buildings_dataset.loc[all_predicted_buildings_dataset.intersects(true_labels_dataset.unary_union)].reset_index(drop=True)

    # Dissolve polygons that belong to the same building in labels, to avoid several counts for the same detection
    true_new_frittliggende_buildings = true_new_frittliggende_buildings.dissolve(by='label_id')

    return true_new_frittliggende_buildings


def get_new_buildings_fasit(true_labels, fkb_labels):
    #Buffer up fkb-labels to avoid getting border of buildings as difference:
    fkb_labels.buffer(0.3)
    true_new_frittliggende_buildings = true_labels.loc[~true_labels.intersects(fkb_labels.unary_union)]
    return true_new_frittliggende_buildings

