# KartAI

[KartAi](https://kartai.no/) is a research project with the objective to use AI to improve the map data for buildings in Norway.

This repository is intended for people contributing and working on the KartAi project, and in order to use the scripts you need access to both image data sources and azure resources.

The repository allows you to easily create training data for any sort of vector data and a set of different image data sources, everything defined in config files passed to the scripts. It also allows for training a set of different implemented Tensorflow models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Conda environment](#conda-environment)
  - [Running scripts](#running-scripts)
  - [Environment variables](#environment-variables)
- [Models](#models)
  - [Implemented models](#implemented-models)
  - [Download existing trained models](#download-existing-trained-models)
  - [Upload model](#upload-model)
- [Training data](#training-data)
  - [Dataset config file](#dataset-config-file)
    - [Tile grid](#tile-grid)
    - [Image Sources](#image-sources)
  - [Create Training Data Script](#create-training-data-script)
  - [Data teacher](#data-teacher)
- [Train](#train)
  - [Datagenerator config file](#datagenerator-config-file)
  - [Train script](#train-script)
  - [Run multiple training processes](#run-multiple-training-processes)
- [Evaluating the models](#evaluating-the-models)
  - [IoU result table](#iou-result-table)
  - [Building count result table](#building-count-result-table)
- [Using the trained models](#using-the-trained-models)
  - [Predict](#predict)
  - [Create vectordata](#create-vectordata)

## Prerequisites

In order to create the training data for building segmentation you need access to a WMS of aerial images, and a database containing existing vector-data of buildings.

## Setup

### Conda environment

To make sure you have correct versions of all packages we recommend using anaconda.

Create a conda environment with python 3.9 by running command below. Replace "env-name" with desired name for the environment

`conda create -n "env-name" python=3.9`

`conda activate "env-name"`

### Running scripts

To run the scripts, we have two options depending on your operating system:
From the project root, run:

Unix: `./kai <args>`

Windows: `kai.bat <args>`

### Environment variables

To run the the program you need a env_sevrets.py file that contains secret enviroment variables, as well as whitelisting your ip-address. Contact a developer for access.

## Models

### Implemented models

Our implemented models are:

- unet
- resnet
- bottleneck
- bottleneck_cross (custom architecture)
- CSP
- CSP_cross (custom architecture)
- unet-twin (custom architecture)

### Download existing trained models

To download the trained models checkpoint files and the metadata files to view hyperparameters and performance you can run:

Unix:

`./kai download_models`

Windows:

`kai.bat download_models`

This will download all available models that are not already downloaded into `/checkpoints` directory.

### Upload model

If a trained model was created but not uploaded to azure automatically (flag when running the training), you can upload the model by running:

Unix:

`./kai upload_model -m {model_name}`

Windows:

`kai.bat upload_model -m {model_name}`

## Training data

Dataset is automatically created based on the given data sources defined in the Dataset config file.

### Dataset config file

The dataset config file (used in the -c argument to create_training_data) is a json file describing the datasets
used for training / validation / test. It has three main sections: `"TileGrid"`, `"ImageSources"` and the
image sets.

Main structure:

```json
{
  "TileGrid": {
    "srid": 25832,
    "x0": 563000.0,
    "y0": 6623000.0,
    "dx": 100.0,
    "dy": 100.0
  },
  "ImageSources": [
    {
       ...
    },
    {
       ...
    }
  ],
  "ImageSets": [
    {
       ...
    },
    {
       ...
    }
  ]
}
```

#### Tile grid

The TileGrid defines the grid structure for the image tiles.

```json
 "TileGrid": {
    "srid": 25832,
    "x0": 410000.0,
    "y0": 6420000.0,
    "dx": 100.0,
    "dy": 100.0
  }
```

All image tiles will be in the spatial reference system given by `"srid"`. The tiles will be of size `dx * dy`, with
tile `(0, 0)` having the lower left corner at `(x0, y0)`, tile `(1, 0)` at `(x0 + dx, y0)` etc...

#### Image Sources

The ImageSources is a list of image sources: database layers, WMS/WCS services, file layers (shape, geojson, ...)

Example of Postgres image datasource:

```json
{
  "name": "BuldingDb",
  "type": "PostgresImageSource",
  "host": "pg.buildingserver.org",
  "port": "5432",
  "database": "Citydatabase",
  "user": "databaseuser",
  "passwd": "MyVerySecretPW",
  "image_format": "image/tiff",
  "table": "citydb.building_polygon"
}
```

Example of WMS image datasource:

```json
{
  "name": "OrtofotoWMS",
  "type": "WMSImageSource",
  "image_format": "image/tiff",
  "url": "https://waapi.webatlas.no/wms-orto/",
  "layers": ["ortofoto"],
  "styles": ["new_up"]
}
```

Example of image source. Note that `"image_format"` is for the format of the output cache-mosaic. The system uses
GDAL for file handling, and all valid GDAL import formats (including .vrt - GDAL Virtual Format) can be read.

```json
{
  "name": "Ortofoto_manual",
  "type": "ImageFileImageSource",
  "image_format": "image/tiff",
  "file_path": "training_data/cityarea/ortofoto/aerialimage.tif"
}
```

Example of vector file source. GDAL / OGR is used for reading image models, and all valid OGR import
formats can be read.

```json
{
  "name": "Building_smallset",
  "type": "VectorFileImageSource",
  "image_format": "image/tiff",
  "file_path": "training_data/cityarea/shape/building.shp",
  "srid": 25832
}
```

Example of project arguments, that will affect production of the dataset.

```json
{
  "ProjectArguments": {
    "training_fraction": 1,
    "validation_fraction": 0,
    "shuffle_data": "True",
    "max_size": 100
  }
}
```

### Created dataset format

Once a dataset is created there will be several files generated.
Labels area created and saved to `training_data/AzureByggDb/{tilegrid}/{tilesize}`.
However, the WMSImageSource are not created yet, only information about how the can be downloaded. The actual data is downloaded once you start training a model with the given dataset.


### Create Training Data Script

Arguments:

| Argument | Description                                                                                                                                                                                      |
| -------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -n       | what to name the dataset                                                                                                                                                                         |
| -c       | path to config file                                                                                                                                                                              |
| --region | Polygon or multipolygon describing data area with coordinates in same system as defined in config (i.e EPSG:25832), WKT or geojson (geometry) format, directly in a text string or as a filename |
| --x_min  | x_min for bbox, alternative to --region                                                                                                                                                          |
| --y_min  | y_min for bbox, alternative to --region                                                                                                                                                          |
| --x_max  | x_max for bbox, alternative to --region                                                                                                                                                          |
| --y_max  | y_max for bbox, alternative to --region                                                                                                                                                          |

Example:

**Unix:**

With bbox: `./kai create_training_data -n medium_area -c config/dataset/bygg.json --x_min 618296.0 --y_min 6668145.0 --x_max 623495.0 --y_max 6672133.0`

With region: `./kai create_training_data -n small_test_area -c config/dataset/bygg.json --region training_data/regions/small_building_region.json`

**Windows:**

With bbox: `kai.bat dataset/create_training_data -n medium_area -c config/dataset/bygg.json --x_min 618296.0 --y_min 6668145.0 --x_max 623495.0 --y_max 6672133.0`

With region: `kai.bat create_training_data -n small_test_area -c config/dataset/bygg.json --region training_data/regions/small_building_region.json`

## Data teacher

Data teacher is a module where training of the model happens in combination with creating a good dataset to train on. It uses the model to find dataset instances that the model needs to learn more from, and adds the areas to the training dataset. This way both the model and the dataset improves in iteration, until training of the model converges (meaning that adding more data doesn't help the model to improve further.)

Arguments:

| Argument | Description                                                                                                                                                                                                                                                    | Required | Default                                   |
| -------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------- | ----------------------------------------- |
| -n       | prefix name for the datasets and models that is created during the iterations                                                                                                                                                                                  | Yes      |
| -v_dn    | name of the dataset to use for validation of the trained model in each data teacher iteration                                                                                                                                                                  | Yes      |
| -t_dn    | name of the init dataset to train the models with                                                                                                                                                                                                              | Yes      |
| -cn      | init checkpoint that is used for first iteration instead of training from scratch                                                                                                                                                                              | No       |
| --region | The region that is used when expanding the training data. Represented as a Polygon or MultiPolygon with coordinates in same system as defined in dataset config (i.e EPSG:25832), WKT or geojson (geometry) format, directly in a text string or as a filename |
| -m       | name of model to train                                                                                                                                                                                                                                         | Yes      |
| -dc      | Path for data dataset config file                                                                                                                                                                                                                              | No       | `config/dataset/bygg_auto_expanding.json` |
| -igc     | Path for the ml-input-generator config file                                                                                                                                                                                                                    | No       | `config/ml_input_generator/ortofoto.json` |
| -f       | number of features                                                                                                                                                                                                                                             | No       | 32                                        |
| -a       | activation function                                                                                                                                                                                                                                            | No       | relu                                      |
| -bs      | batch size                                                                                                                                                                                                                                                     | No       | 8                                         |
| -opt     | Chosen optimizer                                                                                                                                                                                                                                               | No       | RMSprop                                   |
| -test    | Wether to run in test mode, used when looking for bugs                                                                                                                                                                                                         | No       | Falses                                    |

Unix:

`./kai data_teacher -n kystlinje_sornorge -v_dn validation_dataset -t_dn training_datateacher_set -m unet --region training_data/regions/auto_expand_region.json`

Windows:

`kai.bat data_teacher -n kystlinje_sornorge -v_dn validation_dataset -t_dn training_datateacher_set -m unet --region training_data/regions/auto_expand_region.json`

## Train

When training a model you can define both the data to train on, and the model to train. You also need to define a datagenerator config file, that tells the train process how the to feed the model with the given data.

### Datagenerator config file

In order to tell the data generator how to feed the ML-model with data, we have created config files under `config/ml_input_generator` that determines the shape of the inputs.
We can choose between stacking all the inputs into one image with several channels, or we can keep each image as a separate input in a tuple.

```json
{
  "model_input_stack": [
    { "name": "image", "dimensions": [512, 512, 3] },
    { "name": "lidar", "dimensions": [512, 512, 1] }
  ],
  "model_input_tuple": [],
  "ground_truth": { "name": "label", "dimensions": [512, 512, 1] }
}
```

### Train script

Arguments:

| Argument | Description                                                                                     | Required | Default                                   |
| -------- | :---------------------------------------------------------------------------------------------- | :------- | ----------------------------------------- |
| -dn      | name of the dataset to train with (can pass several -dn arguments to train on several datasets) | Yes      |
| -cn      | name of result model (checkpoint name)                                                          | Yes      |
| -m       | name of model to train (see [the list of implemented models](#models))                                                                          | Yes      |
| -c       | Path for data generator config file                                                             | No       | `config/ml_input_generator/ortofoto.json` |
| -s       | Save trained model to azure                                                                     | No       | True                                      |
| -f       | number of features                                                                              | No       | 32                                        |
| -a       | activation function                                                                             | No       | relu                                      |
| -e       | number of epochs to run                                                                         | No       | 100                                       |
| -bs      | batch size                                                                                      | No       | 8                                         |
| -opt     | Chosen optimizer                                                                                | No       | RMSprop                                   |
| -ft      | Checkpoint to finetune model from                                                               | No       | False                                     |

Example:

**Unix:**

Single dataset:

`./kai train -dn {dataset_name} -cn {checkpoint_name} -c{config/ml_input_generator/ortofoto.json} -m {model_name} -a {activation} -bs 4 -f 16 -e {epochs}`

Several datasets:

`./kai train -dn {dataset_name_1} -dn {dataset_name_2} -cn {checkpoint_name} -c{config/ml_input_generator/ortofoto.json} -m {model_name} -a {activation} -bs 4 -f 16 -e {epochs}`

**Windows:**

Single dataset:

`kai.bat train -dn {dataset_name} -cn {checkpoint_name} -c{config/ml_input_generator/ortofoto.json} -m {model_name} -a {activation} -bs 4 -f 16 -e {epochs}`

Several datasets:

`kai.bat train -dn {dataset_name_1} -dn {dataset_name_2} -cn {checkpoint_name} -c{config/ml_input_generator/ortofoto.json} -m {model_name} -a {activation} -bs 4 -f 16 -e {epochs}`

When training is complete, the resulting checkpoint file and metadata file is automatically uploaded to azure.

### Run multiple training processes

In order to test lots of models and hyperparameters we can run the compare_models scripts which will sequentially run as many models as you like. Change the script to run different models with desired hyperparameters, and start script by running:

Unix:

`./kai compare_models -dn {dataset_name}`

Windows:

`kai.bat compare_models -dn {dataset_name}`

## Evaluating the models

We have created an automatic process for generating a result table that gives an overview of all the trained models performance.


### IoU result table

For a complete view of performance of the different models, run:

Unix:

`./kai results`

Windows:

`kai.bat results`

The script opens a GUI table to view results, as well as an excel file.

### [WIP] Building count result table

By adding the parameter `-ksand true` you will instead get a full list of how each model is performing on the given test area for the project.

Unix:

`./kai results -ksand true`

Windows:

`kai.bat results -ksand true`

## Using the trained models

### Predict

Run prediction with one of the trained models.
Running prediction will download to wanted model from azure, before running prediction.

Arguments:

| Argument | Description                                   |
| -------- | :-------------------------------------------- |
| -dn      | name of dataset to predict on                 |
| -cn      | name of the trained model used for prediction |
| -c       | Path for data generator config file           |
| -s       | wether or not to save result images           |
| -ft      | Checkpoint to finetune model from             |

Example:

Unix:

`./kai predict -dn building_dataset -cn unet_model`

Windows:

`kai.bat predict -dn building_dataset -cn unet_model`

### Create vectordata

Create a vector dataset with predicted data from a chosen ML model, on a chosen region.
Running creation of vectordata will download to wanted model from azure, before running prediction.

Arguments:

| Argument |Description  |  type |  required | default|                                                                                                                                                                                     |
| -------- | :--------| -----| -----| -----| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -n       | name of resulting dataset | string | yes                                                                                                                                                                        |
| -cn      | name of the trained model used for prediction   | string | yes                                                                                                                                                 |
| -c       | path to config file         | string | yes                                                                                                                                                                     |
| --region | Polygon or MultiPolygon describing data area with coordinates in same system as defined in config (i.e EPSG:25832), WKT or geojson (geometry) format, directly in a text string or as a filename | WKT, jsontext, or filename | yes |
| -mb      |Max batch size for creating mosaic of the predictions | string | No | 200
| -c      |Data config path | string | yes
| -raw      | Whether to create only raw predictions, or create tilbygg, frittliggende and existing buildings as well | bool | No | True
| -p      | Whether to skip directly to postprocessing, and not look for needed downloaded data. Typically used if you have already run production of dataset for same area, but with different model | bool | No | False
| -s      | Whether to save resulting vectordata to azure or locally. Options as 'local' or 'azure' | string | No | azure

Example:

Unix:

`./kai create_predicted_buildings_dataset -n exiting_dataset -cn unet_model --region training_data/karmoy.json`

Windows:

`kai.bat create_predicted_buildings_dataset -n exiting_dataset -cn unet_model --region training_data/karmoy.json`
