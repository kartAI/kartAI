import os
import uuid
import env
from pathlib import Path
from azure.storage.blob import BlobServiceClient, __version__


def upload_model_to_azure_blobstorage(modelname):

    try:
        print("Azure Blob Storage v" + __version__ +
              " - uploading the trained models to blob")

        # connection string is stored in an environment variable on the machine
        connect_str = env.get_env_variable('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(
            connect_str)

        upload_model_file(modelname, blob_service_client, )
        upload_metadata(modelname, blob_service_client)

    except Exception as ex:
        print('Exception:', ex)


def upload_data_to_azure(data, filename, azure_container_name):
    connect_str = env.get_env_variable('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(
        connect_str)

    dataset_blob_client = blob_service_client.get_blob_client(
        container=azure_container_name, blob=filename)

    dataset_blob_client.upload_blob(data)

    print("\nUploading to Azure Storage as blob:\n\t" + filename)


def upload_model_file(modelname, blob_service_client):
    model_container_name = env.get_env_variable("models_container_name")

    model_file_name = modelname+'.h5'
    model_path = os.path.join(env.get_env_variable(
        'trained_models_directory'), model_file_name)

    model_blob_client = blob_service_client.get_blob_client(
        container=model_container_name, blob=model_file_name)

    with open(model_path, "rb") as model:
        model_blob_client.upload_blob(model)

    print("\nUploading to Azure Storage as blob:\n\t" + model_file_name)


def upload_metadata(modelname, blob_service_client):
    metadata_container_name = env.get_env_variable("metadata_container_name")

    metadata_file_name = modelname+'.meta.json'
    metadata_path = os.path.join(env.get_env_variable(
        'trained_models_directory'), metadata_file_name)

    metadata_blob_client = blob_service_client.get_blob_client(
        container=metadata_container_name, blob=metadata_file_name)

    with open(metadata_path, "rb") as metadata:
        metadata_blob_client.upload_blob(metadata)

    print("\nUploading to Azure Storage as blob:\n\t" + metadata_file_name)


def upload_model_performance_file(performance_file_name, region_name):
    connect_str = env.get_env_variable('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(
        connect_str)
    performance_container_name = env.get_env_variable(
        f"{region_name}_performances_container_name")

    metadata_path = os.path.join(env.get_env_variable(
        'prediction_results_directory'), f'{region_name}_performance/{performance_file_name}.json')

    metadata_blob_client = blob_service_client.get_blob_client(
        container=performance_container_name, blob=performance_file_name+'.json')

    with open(metadata_path, "rb") as metadata:
        metadata_blob_client.upload_blob(metadata)

    print("\nUploading to Azure Storage as blob:\n\t" + performance_file_name)


def get_available_trained_models():
    try:
        connect_str = env.get_env_variable('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(
            connect_str)

        container_name = env.get_env_variable("models_container_name")
        container_client = blob_service_client.get_container_client(
            container_name)
        availableModels = container_client.list_blobs()

        model_names = [sub['name'] for sub in availableModels]
        return model_names

    except Exception as ex:
        print('Exception:', ex)
        raise Exception("Could not fetch existing trained models")


def get_available_performances(region_name):
    try:
        connect_str = env.get_env_variable('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(
            connect_str)

        container_name = env.get_env_variable(
            f"{region_name}_performances_container_name")
        container_client = blob_service_client.get_container_client(
            container_name)
        availablePerformanceFiles = container_client.list_blobs()

        performances_names = [sub['name'] for sub in availablePerformanceFiles]
        return performances_names

    except Exception as ex:
        print('Exception:', ex)
        raise Exception("Could not fetch existing trained models")


def download_trained_models():
    models = get_available_trained_models()
    for model in models:
        checkpoint_path = os.path.join(env.get_env_variable(
            'trained_models_directory'), model)
        if not os.path.isfile(checkpoint_path):
            print('\nDownloading: ', model)
            download_model_file_from_azure(Path(model).stem)


def download_performances(download_file_path, region_name):
    performance_metafiles = get_available_performances(region_name)
    for performance_file in performance_metafiles:
        performance_file_path = os.path.join(
            download_file_path, performance_file)
        if not os.path.isfile(performance_file_path):
            print('\nDownloading: ', performance_file)
            download_performance_file_from_azure(
                Path(performance_file).stem, download_file_path, region_name)


def download_performance_file_from_azure(performance_file_name, download_file_path, region_name):
    try:
        print('\nDownloading trained model')
        connect_str = env.get_env_variable('AZURE_STORAGE_CONNECTION_STRING')

        blob_service_client = BlobServiceClient.from_connection_string(
            connect_str)

        if not os.path.isdir(download_file_path):
            os.mkdir(download_file_path)

        performances_container_name = env.get_env_variable(
            f"{region_name}_performances_container_name")

        model_blob_client = blob_service_client.get_blob_client(
            container=performances_container_name, blob=performance_file_name+'.json')

        with open(os.path.join(download_file_path, performance_file_name+'.json'), "wb") as download_file:
            download_file.write(model_blob_client.download_blob().readall())

    except Exception as ex:
        print('Exception:', ex)


def download_model_file_from_azure(trainedModelName):
    try:
        print('\nDownloading trained model')
        connect_str = env.get_env_variable('AZURE_STORAGE_CONNECTION_STRING')

        blob_service_client = BlobServiceClient.from_connection_string(
            connect_str)

        download_file_path = env.get_env_variable(
            'trained_models_directory')
        if not os.path.isdir(download_file_path):
            os.mkdir(download_file_path)

        model_container_name = env.get_env_variable("models_container_name")

        model_blob_client = blob_service_client.get_blob_client(
            container=model_container_name, blob=trainedModelName+'.h5')

        with open(os.path.join(download_file_path, trainedModelName+'.h5'), "wb") as download_file:
            download_file.write(model_blob_client.download_blob().readall())

        metdata_container_name = env.get_env_variable(
            "metadata_container_name")

        metadata_blob_client = blob_service_client.get_blob_client(
            container=metdata_container_name, blob=trainedModelName+'.meta.json')

        with open(os.path.join(download_file_path, trainedModelName+'.meta.json'), "wb") as download_file:
            download_file.write(metadata_blob_client.download_blob().readall())

    except Exception as ex:
        print('Exception:', ex)
