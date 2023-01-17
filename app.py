#!flask/bin/python
import os
from pathlib import Path

from azure import blobstorage
from flask import Flask, jsonify, request
from flask_cors import CORS

from osgeo import ogr, osr

import env
from kartai.dataset.create_building_dataset import create_predicted_buildings_dataset


def transform(geom, inSpatialNum, outSpatialNum):
    """ Transform the coordinate system of geom in place. """
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inSpatialNum)

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outSpatialNum)

    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    geom.Transform(coordTrans)


def predict_area(req):
    """
    Return the final prediction

    Keyword arguments:
    req (dictionary) -- a dictionary representing a GeoJSON. Must be in the format specified at the top of this file.
    """

    if "geometry" not in req:
        poly = req['area']
        checkpoint = req['model']
    # The coordinates in the query from the client are expected to be in (long, lat).
    # Reorder them as (lat, long) to be easier to work with internally
    coordinates = poly["coordinates"][0]
    poly["coordinates"] = [[list(reversed(x)) for x in coordinates]]

    # Convert the preprocessed query polygon so that it is easier to work with internally
    geom = ogr.CreateGeometryFromJson(str(poly))

    inSpatialNum = 4326
    outSpatialNum = 25832

    # Transform geom to a coordinate system that corresponds to the file system structure used
    transform(geom, inSpatialNum, outSpatialNum)

    model_name = Path(checkpoint).stem
    config_path = "config/dataset/bygg-no-rules.json"

    all_predicted_buildings_dataset = create_predicted_buildings_dataset(
        geom, model_name, config_path)

    # Transform geom to a coordinate system that corresponds to the file system structure used
    all_predicted_buildings_dataset = transform(
        all_predicted_buildings_dataset, outSpatialNum, inSpatialNum)

    return all_predicted_buildings_dataset.to_json()


app = Flask(__name__)
cors = CORS(app)


@app.route("/Models",  methods=['GET'])
def getModels():
    models = blobstorage.getAvailableTrainedModels()
    return jsonify(models)


@app.route("/Datasets",  methods=['GET'])
def getAvailableDatasets(self):
    # Find all available checkpoint files
    training_data_dir = env.get_env_variable('created_datasets_directory')
    availableDatasetFilePaths = os.listdir(training_data_dir)
    datasetNames = availableDatasetFilePaths

    return jsonify(datasetNames)


@app.route("/Prediction",  methods=['POST'])
def analyse_area():
    """POST method so that the client can POST a polygon and receive a GeoJSON that contains
    predictions from within this polygon"""
    req = request.get_json(force=True)
    response = predict_area(req)
    return response
