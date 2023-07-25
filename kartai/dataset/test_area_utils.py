import os
import env


def get_test_region_avgrensning_dir(region_name):
    cached_data_dir = env.get_env_variable('cached_data_directory')
    if region_name == "ksand":
        return os.path.join(cached_data_dir, "AP2_T2_geodata/Prosjektområde/shape/avgrensning.shp")
    elif region_name == "balsfjord":
        return os.path.join(cached_data_dir, "balsfjord_test_area/balsfjord_test_area_avgrensning.shp")
    else:
        raise NotImplementedError(
            "Supports only region ksand and balsfjord, not region", region_name)


def get_label_files_dir_for_test_region(region_name):

    cached_data_dir = env.get_env_variable(
        'cached_data_directory')

    if(region_name == "ksand"):
        label_bygning_path = os.path.join(
            cached_data_dir, "AP2_T2_geodata/Prosjektområde/shape/bygning.shp")
        label_tiltak_path = os.path.join(
            cached_data_dir, "AP2_T2_geodata/Prosjektområde/shape/tiltak.shp")
        return [label_bygning_path, label_tiltak_path]
    elif(region_name == "balsfjord"):
        return [os.path.join(cached_data_dir, "balsfjord_test_area/balsfjord_test_area_labels_adjusted.shp")]
    else:
        raise NotImplementedError(
            "Supports only region ksand and balsfjord, not region: ", region_name)
