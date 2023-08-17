import json


def read_config(config_path: str) -> dict:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config
