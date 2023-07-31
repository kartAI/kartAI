import os
import geopandas as gp
import rasterio.merge
from kartai.dataset.Iou_calculations import get_geo_data_frame
from kartai.dataset.create_building_dataset import get_raw_predictions, get_valid_geoms, perform_last_adjustments
from kartai.dataset.test_area_utils import get_adjusted_labels_dir, get_test_region_avgrensning_dir


def get_performance_count_for_detected_buildings(all_predicted_buildings_dataset, predictions_path, true_labels, fkb_labels, new_buildings_fasit, CRS_prosjektomrade, model_name, performance_output_dir, region_name):
    
    predicted_new_buildings = get_predicted_new_buildings(
        all_predicted_buildings_dataset, fkb_labels, predictions_path, CRS_prosjektomrade)

    correctly_detected_buildings = get_correctly_detected_buildings(
        true_labels, all_predicted_buildings_dataset)

    correctly_detected_new_buildings = get_correctly_detected_new_buildings(
        true_labels, all_predicted_buildings_dataset)

    wrongly_detected_buildings = get_wrongly_detected_frittliggende_bygg_dataset(
        predicted_new_buildings, new_buildings_fasit)

    # Looking only at the ones missing in FKB
    missing_new_frittliggende_buildings = get_FKB_missing_frittliggende_buildings(
        new_buildings_fasit, all_predicted_buildings_dataset)

    # Looking at all existing buildings not detected
    missing_frittliggende_buildings = get_all_missing_frittliggende_buildings(
        true_labels, all_predicted_buildings_dataset)

    # Cut all datasets to test-region - the fetched prediction tiles cover area outside as well
    area_to_predict = get_test_region_avgrensning_dir(region_name) #TODO: use region instead
    
    area_gdf = gp.read_file(area_to_predict)
    wrongly_detected_buildings = gp.clip(
        wrongly_detected_buildings, area_gdf)
    all_predicted_buildings_dataset = gp.clip(
        all_predicted_buildings_dataset, area_gdf)

    save_dataset(missing_new_frittliggende_buildings,
                 performance_output_dir,  model_name+'_missing_new_buildings')
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

    return get_dataset_count(wrongly_detected_buildings), get_dataset_count(correctly_detected_buildings),  get_dataset_count(correctly_detected_new_buildings), get_dataset_count(missing_new_frittliggende_buildings), get_dataset_count(missing_frittliggende_buildings)


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


def get_FKB_missing_frittliggende_buildings(true_new_FKB_missing_frittliggende_buildings, all_predicted_buildings_dataset):
    # Find missing buildings
    not_detected_frittliggende_area = gp.overlay(
        true_new_FKB_missing_frittliggende_buildings, all_predicted_buildings_dataset, how='difference')

    not_detected_frittliggende_area.index = not_detected_frittliggende_area['label_id']

    not_detected_frittliggende_area['rest_area_after_label_diff'] = not_detected_frittliggende_area.geometry.area - \
        true_new_FKB_missing_frittliggende_buildings.geometry.area

    missing_frittliggende_buildings = not_detected_frittliggende_area.loc[
        not_detected_frittliggende_area['rest_area_after_label_diff'] == 0]
    return missing_frittliggende_buildings


def get_all_missing_frittliggende_buildings(true_labels, all_predicted_buildings_dataset):
    not_detected_frittliggende_area = gp.overlay(
        true_labels, all_predicted_buildings_dataset, how='difference')

    not_detected_frittliggende_area.index = not_detected_frittliggende_area['label_id']

    not_detected_frittliggende_area['rest_area_after_label_diff'] = true_labels.geometry.area - \
        not_detected_frittliggende_area.geometry.area

    missing_frittliggende_buildings = not_detected_frittliggende_area
    missing_frittliggende_buildings = missing_frittliggende_buildings.loc[
        not_detected_frittliggende_area['rest_area_after_label_diff'] == 0]

    return missing_frittliggende_buildings


def get_wrongly_detected_frittliggende_bygg_dataset(predicted_new_frittliggende_bygg_dataset, true_new_frittliggende_buildings):
    # Find false detected buildings
    detected_buildings_minus_existing_buildings = gp.overlay(
        predicted_new_frittliggende_bygg_dataset, true_new_frittliggende_buildings, how='difference')

    detected_buildings_minus_existing_buildings.index = detected_buildings_minus_existing_buildings[
        'b_id']

    detected_buildings_minus_existing_buildings['rest_area_after_label_diff'] = \
        detected_buildings_minus_existing_buildings.geometry.area - \
        predicted_new_frittliggende_bygg_dataset.geometry.area

    false_detected_buildings = detected_buildings_minus_existing_buildings.loc[
        detected_buildings_minus_existing_buildings['rest_area_after_label_diff'] == 0]
    return false_detected_buildings


def get_correctly_detected_new_buildings(true_new_frittliggende_buildings, predicted_new_frittliggende_bygg_dataset):
    # Find true detected buildings
    true_detected_buildings = predicted_new_frittliggende_bygg_dataset.loc[~predicted_new_frittliggende_bygg_dataset.intersects(true_new_frittliggende_buildings.unary_union)]

    # Dissolve polygons that belong to the same building in labels, to avoid several counts for the same detection
    true_detected_buildings = true_detected_buildings.dissolve(by='label_id')
    return true_detected_buildings


def get_predicted_new_buildings(all_predicted_buildings_dataset, fkb_labels, predictions_path, crs):
    """All predictions that are not in the labels. This returns a set of true and false positives, 
    meaning predictions that area in fact buldings, but also predictions that are wrongly detected as buildings"""

    #Get all predictions that doesn't overlap existing labels
    predicted_buildings_dataset = all_predicted_buildings_dataset.loc[~all_predicted_buildings_dataset.intersects(fkb_labels.unary_union)]

    raw_prediction_imgs = get_raw_predictions(predictions_path)
    full_img, full_transform = rasterio.merge.merge(raw_prediction_imgs)

    # Adding prob value, removing small buildings ++
    predicted_buildings_dataset = perform_last_adjustments(
        predicted_buildings_dataset, full_img, full_transform, crs)
    predicted_buildings_dataset.index = predicted_buildings_dataset[
        'b_id']
    return predicted_buildings_dataset


def get_true_labels(region_name, CRS_prosjektomrade):
    true_labels_dataset = get_adjusted_labels(
        region_name, CRS_prosjektomrade)
    # Adding value to dissolve by, want to dissolve all buildings
    true_labels_dataset['value'] = 1
    # Buffer up a little bit in order to merge buildings that are barely not connected, but that should be considered as one building
    true_labels_dataset['geometry'] = true_labels_dataset.buffer(0)
    true_labels_dataset['label_id'] = true_labels_dataset.index
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
    true_labels_minus_predictions = gp.overlay(
        true_labels_dataset, all_predicted_buildings_dataset, how='difference')

    true_labels_minus_predictions.index = true_labels_minus_predictions['label_id']

    true_labels_minus_predictions['rest_area_after_diff'] = true_labels_dataset.geometry.area - \
        true_labels_minus_predictions.geometry.area

    true_new_frittliggende_buildings = true_labels_minus_predictions.loc[
        true_labels_minus_predictions['rest_area_after_diff'] != 0]

    return true_new_frittliggende_buildings


def get_new_buildings_fasit(true_labels, fkb_labels):
    #Buffer up fkb-labels to avoid getting border of buildings as difference:
    fkb_labels.buffer(0.3)
    true_new_frittliggende_buildings = true_labels.loc[~true_labels.intersects(fkb_labels.unary_union)]
    return true_new_frittliggende_buildings

