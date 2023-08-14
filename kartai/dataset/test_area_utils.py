import os
import env
from kartai.dataset.resultRegion import ResultRegion


def get_test_region_avgrensning_dir(region_name):
    cached_data_dir = env.get_env_variable('cached_data_directory')
    if ResultRegion.from_str(region_name) == ResultRegion.KRISTIANSAND:
        return os.path.join(cached_data_dir, "AP2_T2_geodata/Prosjektområde/shape/avgrensning.shp")
    elif ResultRegion.from_str(region_name) == ResultRegion.BALSFJORD:
        return os.path.join(cached_data_dir, "regions/balsfjord_test_area.geojson")
    else:
        raise NotImplementedError(
            "Supports only region ksand and balsfjord, not region", region_name)


def get_adjusted_labels_dirs(region_name):

    cached_data_dir = env.get_env_variable(
        'cached_data_directory')

    if ResultRegion.from_str(region_name) == ResultRegion.KRISTIANSAND:
        label_bygning_path = os.path.join(
            cached_data_dir, "AP2_T2_geodata/Prosjektområde/shape/bygning.shp")
        label_tiltak_path = os.path.join(
            cached_data_dir, "AP2_T2_geodata/Prosjektområde/shape/tiltak.shp")
        return [label_bygning_path, label_tiltak_path]
    elif ResultRegion.from_str(region_name) == ResultRegion.BALSFJORD:
        return [os.path.join(cached_data_dir, "balsfjord_test_area/adjusted_labels.shp")]
    else:
        raise NotImplementedError(
            "Supports only region ksand and balsfjord, not region: ", region_name)
