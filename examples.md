### Examples of scripts to run

data_generators:

- height_orto_stack.json
- height_orto_tuple.json
- ortofoto.json

### Predict

./kai predict -dn hoyde_data -cn unet_twin -c config/ml_input_generator/height_orto_tuple.json
./kai predict -dn hoyde_data -cn ksand_unet_twin -c config/ml_input_generator/height_orto_tuple.json
./kai predict -dn ksand_with_hoyde -cn ksand_unet_4channels_100_epochs_mish -c config/ml_input_generator/height_orto_stack.json
./kai predict -dn building_area -cn unet_basic -c config/ml_input_generator/ortofoto.json
./kai predict -dn building_area -cn bottleneck_cross_net_relu_f32_d4 -c config/ml_input_generator/ortofoto.json

./kai predict -dn building_area -cn kystlinje_sornorge_data_teacher_init -c config/ml_input_generator/ortofoto.json

### On mac

#### Existing datasets:

- dummy
- hoyde_data
- kristiansand_manually_adjusted
- ksand_with_hoyde

#### Create training data:

./kai create_training_data -n hoyde_ortofoto_medium_area -c config/dataset/bygg_ndh.json --x_min 618296.0 --y_min 6668145.0 --x_max 623495.0 --y_max 6672133.0
./kai create_training_data -n ksand -c config/dataset/ksand-manuell.json --x_min 437300 --y_min 6442000 --x_max 445700 --y_max 6447400

./kai create_training_data -n ksand_manuelt -c config/dataset/ksand-manuell.json --x_min 437300 --y_min 6442000 --x_max 445700 --y_max 6447400

./kai create_training_data -n ksand_with_hoyde -c config/dataset/ksand-manuell_with_hoyde.json --x_min 437300 --y_min 6442000 --x_max 445700 --y_max 6447400
./kai create_training_data -n regions_bygg -c config/dataset/bygg.json --region training_data/regions_multi_polygon.json
./kai create_training_data -n karmoy -c config/dataset/bygg.json --region training_data/regions/karmoy.json
./kai create_training_data -n karmoy -c config/dataset/bygg.json --region training_data/regions/karmoy.json

./kai create_training_data -n test_azure -c config/dataset/bygg.json --x_min 437300 --y_min 6442000 --x_max 437310 --y_max 644300
./kai create_training_data -n building_area -c config/dataset/bygg.json --x_min 563140.1 --y_min 6623967.7 --x_max 565757.0 --y_max 6625438.3

Creating dataset with images with bad predictions:
./kai create_training_data -n auto_expanded -c config/dataset/bygg_auto_expanding.json --x_min 563140.1 --y_min 6623967.7 --x_max 565757.0 --y_max 6625438.3
./kai create_training_data -n building_area_limit -c config/dataset/bygg_limit.json --region training_data/regions/karmoy.json
./kai create_training_data -n building_area_limit_debug_noshuffle -c config/dataset/bygg_limit.json --region training_data/regions/small_building_region.json
./kai create_training_data -n building_area_limit_debug_shuffle -c config/dataset/bygg_limit.json --region training_data/regions/small_building_region.json

./kai create_training_data -n large_building_area -c config/dataset/bygg.json --region training_data/regions/large_building_area.json
./kai create_training_data -n extra_large_building_area -c config/dataset/bygg.json --region training_data/regions/extra_large_building_area.geojson

./kai create_training_data -n large_building_area_ndh -c config/dataset/bygg_ndh.json --region training_data/regions/large_building_area.json

Create validation dataset for data teacher
./kai create_training_data -n validation_dataset -c config/dataset/bygg-no-rules.json --region training_data/regions/validation_region.json

Create training data to init data teacher
./kai create_training_data -n training_datateacher_set -c config/dataset/bygg.json --region training_data/regions/auto_expand_training_region.json

Ksand not adjusted - add all data to test-set to be able to predict entire area:
./kai create_training_data -n ksand_prosjektomrade_not_adjusted_test_set -c config/dataset/bygg-no-rules-all-test.json --region training_data/regions/prosjektomr_test.json
m hoyde

./kai create_training_data -n ksand_prosjektomrade_not_adjusted -c config/dataset/bygg-no-rules-all-test.json --region training_data/regions/prosjektomr_test.json

./kai create_training_data -n ksand_ndh_prosjektomrade_not_adjusted_test_set -c config/dataset/bygg_ndh_no_rules_all_test.json --region training_data/regions/prosjektomr_test.json


./kai create_training_data -n grimstad_municipality -c config/dataset/bygg.json --region training_data/regions/grimstad_municipality.json
./kai create_training_data -n grimstad_municipality -c config/dataset/bygg-no-rules.json --region training_data/regions/grimstad_municipality.json






#### Create predicted building dataset:

./kai create_predicted_buildings_dataset -n ksand_test --region training_data/regions/small_test_region.json -cn CSP_d4_mish -c config/dataset/bygg-no-rules.json
./kai create_predicted_buildings_dataset -n karmoy_nrk_analyse --region training_data/regions/small_test_region.json -cn CSP_d4_mish -c config/dataset/bygg-no-rules.json
./kai create_predicted_buildings_dataset -n karmoy_nrk_analyse --region training_data/regions/karmoy.json -cn CSP_d4_mish -c config/dataset/bygg-no-rules.json
./kai create_predicted_buildings_dataset -n karmoy_nrk_analyse_finetuned --region training_data/regions/karmoy.json -cn CSP_d4_mish_finetuned_karmoy_finetuned_50e -c config/dataset/bygg-no-rules.json

./kai create_predicted_buildings_dataset -n building_area --region training_data/regions/validation_region.json -cn resnet_swish_building_area_d5 -c config/dataset/bygg-no-rules.json

Height model:
./kai create_predicted_buildings_dataset -n ksand_hoyde_model --region training_data/regions/prosjektomr_test.json -cn unet_height_model -c config/dataset/bygg_ndh.json

GRIMSTAD:
./kai create_predicted_buildings_dataset -n grimstad_partly --region training_data/regions/grimstad.geojson -cn CSP_building_area_d3_swish -c config/dataset/bygg-no-rules.json


./kai create_predicted_buildings_dataset -n grimstad_partly --region training_data/regions/grimstad.geojson -cn extra_large_building_area_unet -c config/dataset/bygg-no-rules.json

./kai create_predicted_buildings_dataset -n grimstad_partly --region training_data/regions/grimstad.geojson -cn finetuned_grimstad_resnet_mish_sorlandet_data_teacher_5_v2 -c config/dataset/bygg-no-rules.json

Best checkpoint:
unet_large_building_area_d4_mish_focal_ndh

./kai create_predicted_buildings_dataset -n ksand_hoyde_model --region training_data/regions/prosjektomr_test.json -cn unet_large_building_area_d4_mish_focal_ndh -c config/dataset/bygg-ndh-no-rules.json

./kai create_predicted_buildings_dataset -n CSP_large_building_area_d4_mish --region training_data/regions/prosjektomr_test.json -cn CSP_large_building_area_d4_mish -c config/dataset/bygg-no-rules.json

./kai create_predicted_buildings_dataset -n grimstad_xl_resnet_mish --region training_data/regions/grimstad.geojson -cn xl_resnet_mish_2 -c config/dataset/bygg-no-rules.json

./kai create_predicted_buildings_dataset -n arendal_xl_resnet_mish --region training_data/regions/ArendalBorderCropped.geojson -cn xl_resnet_mish_2 -c config/dataset/bygg-no-rules.json


./kai create_predicted_buildings_dataset -n ringebu_xl_resnet_mish --region training_data/regions/ringebu.geojson -cn xl_resnet_mish_2 -c config/dataset/bygg-no-rules.json -raw true

./kai create_predicted_buildings_dataset -n arendal_xl_csp_mish --region training_data/regions/ArendalBorderCropped.geojson -cn xl_csp_mish -c config/dataset/bygg-no-rules.json -p True -s azure


TESTING:

./kai create_predicted_buildings_dataset -n arendal_xl_resnet_mish --region training_data/regions/ArendalBorderCropped.geojson -cn xl_resnet_mish_2 -c config/dataset/bygg-no-rules.json -p true -an arendal

#### Train:

./kai train -dn kristiansand_manually_adjusted -cn {name} -m unet -f 16 -d 4 -e 50
./kai train -dn dummy -m unet -cn {name} -c config/ml_input_generator/ortofoto.json
./kai train -dn building_area -m unet -cn test_model -c config/ml_input_generator/ortofoto.json
./kai train -dn test_azure -m unet -cn test_model -c config/ml_input_generator/ortofoto.json
./kai train -dn building_area_limit -m unet -cn test_model -c config/ml_input_generator/ortofoto.json -e 2

Train on data from data teacher
./kai train -dn full_analysis_unet_v1_data_teacher_1 -dn full_analysis_unet_v1_data_teacher_0 -dn init_training_data_full_analysis_unet_v1 -m unet -cn unet_data_teacher_datasets -c config/ml_input_generator/ortofoto.json

Several datasets as input:
./kai train -dn test_azure -dn regions_bygg -m unet -cn test_model -c config/ml_input_generator/ortofoto.json

Train on datateacher datasets, and ksand area:
./kai train -dn kristiansand_manually_adjusted -dn kystlinje_sornorge_unet_data_teacher_0 -dn kystlinje_sornorge_unet_data_teacher_1 -dn kystlinje_sornorge_unet_data_teacher_2 -m CSP -cn CSP_d4_mish_datateacher_set -c config/ml_input_generator/ortofoto.json -b 4 -f 16 -d 4 -a mish

Train on datateacher datasets sorlandet, and building area:
./kai train -dn building_area -dn kystlinje_sornorge_unet_data_teacher_0 -dn kystlinje_sornorge_unet_data_teacher_1 -dn kystlinje_sornorge_unet_data_teacher_2 -m CSP_cross_SPP -cn CSP_cross_SPP_buildingarea_sorlandet_datateacher_set -c config/ml_input_generator/ortofoto.json -b 4 -f 16 -d 4 -a relu

Finetune on ksand:

./kai train -dn kristiansand_manually_adjusted -m resnet -cn finetuned_ksand_resnet_large_building_area_d4_mish_focal -ft resnet_large_building_area_d4_mish_focal -b 4 -d 4 -f 16 -a mish -c config/ml_input_generator/ortofoto.json

./kai train -dn kristiansand_manually_adjusted -m CSP -cn finetuned_CSP_large_building_area_d4_mish -ft CSP_large_building_area_d4_mish -b 4 -d 4 -f 16 -a mish -c config/ml_input_generator/ortofoto.json

./kai train -dn kristiansand_manually_adjusted -m CSP -cn finetuned_CSP_large_building_area_d3_swish -ft CSP_large_building_area_d3_swish -b 4 -d 3 -f 16 -a swish -c config/ml_input_generator/ortofoto.json

./kai train -dn kristiansand_manually_adjusted -m resnet -cn finetuned_resnet_mish_sorlandet_data_teacher_1 -ft resnet_mish_sorlandet_data_teacher_1 -b 8 -d 4 -f 16 -a mish -c config/ml_input_generator/ortofoto.json


./kai train -dn large_building_area -m resnet -cn resnet_testrun -b 8 -d 4 -f 16 -a mish -c config/ml_input_generator/ortofoto.json

./kai train -dn large_building_area_ndh -m resnet -cn finetune_large_ndh_xl_resnet_mish -ft xl_resnet_mish  -b 8 -d 4 -f 16 -a mish -c config/ml_input_generator/height_orto_stack.json


./kai train -dn extra_large_building_area -m resnet -cn xl_resnet_mish -b 8 -d 4 -f 16 -a mish -c config/ml_input_generator/ortofoto.json

./kai train -dn extra_large_building_area -m CSP -cn xl_csp_mish -b 6 -d 4 -f 16 -a mish -c config/ml_input_generator/ortofoto.json


./kai train -dn extra_large_building_area -m CSP_cross_SPP -cn xl_csp_cross_mish -b 4 -d 4 -f 8 -a mish -c config/ml_input_generator/ortofoto.json



**Finetune on grimstad**


./kai train -dn grimstad_municipality -m resnet -cn finetuned_grimstad_resnet_mish_sorlandet_data_teacher_5 -ft resnet_mish_sorlandet_data_teacher_5 -b 8 -d 4 -f 16 -a mish -c config/ml_input_generator/ortofoto.json

**Regular train grimstad**
./kai train -dn grimstad_municipality -m resnet -cn grimstad_resnet_mish -b 8 -d 4 -f 16 -a mish -c config/ml_input_generator/ortofoto.json

**Extra large building area**
./kai train -dn extra_large_building_area -m unet -cn extra_large_building_area_unet -b 8 -d 4 -f 16 -a mish -l focal_loss -c config/ml_input_generator/ortofoto.json

### Data teacher

./kai data_teacher -n test -v_dn building_area -m unet --region training_data/regions/karmoy.json

./kai data_teacher -n test -v_dn building_area -t_dn building_area -m unet --region training_data/regions/small_building_region.json

./kai data_teacher -n resnet_kystlinje_sornorge -v_dn validation_dataset -t_dn training_datateacher_set -m unet --region training_data/regions/auto_expand_region.json -m resnet -a mish -d 4 -b 4 -f 16

./kai data_teacher -n kystlinje_sornorge_bygg_ndh -cn kystlinje_sornorge_bygg_ndh_data_teacher_init -th 0.93 -v_dn validation_dataset_bygg_ndh -dc config/dataset/bygg_ndh_auto_expanding.json -igc config/ml_input_generator/height_orto_stack.json -t_dn training_datateacher_bygg_ndh_set -m unet --region training_data/regions/auto_expand_region.json -a mish -d 4 -bs 8 -f 16

./kai data_teacher -n full_analysis_unet_v1 -cn full_analysis_unet_v1_data_teacher_init -v_dn validation_data_full_analysis_unet_v1 -t_dn init_training_data_full_analysis_unet_v1 -m unet --region training_data/regions/auto_expand_region.json -a mish -d 4 -bs 8 -f 16

With ksand as validation area:

./kai data_teacher -n resnet_mish_sorlandet -cn resnet_large_building_area_d4_mish -v_dn kristiansand_manually_adjusted -t_dn large_building_area --region training_data/regions/auto_expand_region.json -a mish -d 4 -bs 8 -f 16 -m resnet

### Full analysis

Beistet:
./kai full_analysis -n resnet_kystlinje_sornorge -m resnet --region_expand training_data/regions/sorlandet_auto_expand_region.json --region_validation training_data/regions/validation_region.json --region_init training_data/regions/auto_expand_training_region.json -a mish -d 4 -bs 6 -f 16

./kai full_analysis -n full_analysis_unet_v1 -m unet --region_expand training_data/regions/sorlandet_auto_expand_region.json --region_validation training_data/regions/validation_region.json --region_init training_data/regions/auto_expand_training_region.json -a swish -d 4 -bs 6 -f 16

Mathilde lokalt:
./kai full_analysis -n test_full_analysis -m bottleneck_cross_SPP --region_expand training_data/regions/auto_expand_region.json --region_validation training_data/regions/validation_region.json --region_init training_data/regions/small_building_region.json -a swish -d 4 -bs 6 -f 16

init checkpoint: full_analysis_unet_data_teacher_init

### Beistet

Existing datasets:

- ksand_with_hoyde




Demo:

./kai create_training_data -n small_test_area -c config/dataset/bygg.json --region training_data/regions/small_building_region.json
./kai train -dn small_test_area -m unet -cn test_small_area_unet -c config/ml_input_generator/ortofoto.json


Nedlasting
./kai download_models

Result table:

./kai results


Produce vector data:

./kai create_predicted_buildings_dataset -n arendal_xl_resnet_mish --region training_data/regions/ArendalBorderCropped.geojson -cn xl_resnet_mish_2 -c config/dataset/bygg-no-rules.json -an arendal

./kai create_predicted_buildings_dataset -n test-region_xl_resnet_mish_2 --region training_data/regions/small_test_region.json -cn xl_resnet_mish_2 -c config/dataset/bygg-no-rules.json -an small-test-region