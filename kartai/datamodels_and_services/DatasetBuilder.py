import queue

from osgeo import gdal
import numpy as np
import os
import json
from kartai.utils.model_utils import model_is_confident
import random
import sys
from kartai.datamodels_and_services.ImageSourceServices import Tile
import multiprocessing as mp


class DatasetBuilder:
    """Class for building datasets and populating caches.

    # The DatasetBuilder is initialized with a dictionary of ImageSets:
    photo_set = ...
    label_set = ...

    dataset_builder = DatasetBuilder({"input": photo_set, "output": label_set})

    # Generate data in a rectangular area
    area = (428000, 6362000, 429000, 6364000)
    dataset = list(dataset_builder.assemble_data(area))

    # The dataset is a list of file paths for each of the image sets
    [{"input": "cache_root/photo_set/25832_450000_6200000_100_100/512/32_74.tif",
      "output": "cache_root/label_set/25832_450000_6200000_100_100/512/32_74.tif"},
      "input": "cache_root/photo_set/25832_450000_6200000_100_100/512/33_74.tif",
      "output": "cache_root/label_set/25832_450000_6200000_100_100/512/33_74.tif"},
      .....
      ]

    """

    def __init__(self, image_set_dict, source_config, project_config=None, confidence_threshold=None,
                 eval_model_checkpoint=None, eager_load=False, num_processes=None):
        """Constructor for DatasetBuilder.
        Takes a dictionary of ImageSets. The dictionary keys are used for indexing data samples in the resulting
        data sample list"""
        self.image_set_dict = {}
        self.tile_grid = None

        if not image_set_dict:
            raise ValueError("no ImageSets")

        image_set_iter = iter(image_set_dict)
        img_set_name = next(image_set_iter)
        img_set = image_set_dict[img_set_name]

        self.tile_grid = img_set.image_source.tile_grid
        if not self.tile_grid:
            raise ValueError("No tile_grid in first image set")
        self.image_set_dict[img_set_name] = img_set

        for img_set_name in image_set_iter:
            img_set = image_set_dict[img_set_name]
            if img_set.image_source.tile_grid is not self.tile_grid:
                raise ValueError("Image sets does not have the same tile_grid")
            self.image_set_dict[img_set_name] = img_set

        self.has_model_confidence_rule = False
        for img_set_name, img_set in self.image_set_dict.items():
            if(img_set.rule and img_set.rule["type"] == "ModelConfidence"):
                self.has_model_confidence_rule = True

        self.source_config = source_config

        self.max_size = None
        if project_config and "max_size" in project_config:
            self.max_size = int(project_config["max_size"])

        self.shuffle_data = \
            project_config and "shuffle_data" in project_config and project_config[
                "shuffle_data"] == "True"

        self.num_processes = num_processes
        self.confidence_threshold = confidence_threshold
        self.eval_model_checkpoint = eval_model_checkpoint
        self.eager_load = eager_load
        if self.num_processes:
            self.eager_load = True

    def _evaluate_rule(self, img, rule):
        np_band = img.array
        if np_band is None:
            print(f"Bilde {img.file_path} ikke lastet")
            return

        if rule["type"] == "And":
            for r in rule["rules"]:
                if not self._evaluate_rule(img, r):
                    return False
            return True

        elif rule["type"] == "Or":
            for r in rule["rules"]:
                if self._evaluate_rule(img, r):
                    return True
            return False

        if rule["type"] == "PixelValueAreaFraction":
            frac_sum = 0
            if np_band.dtype == np.dtype('b') and "values" in rule:

                has_value = False
                for v in rule['values']:
                    if v in np_band:
                        has_value = True

                if not has_value and "more_than" in rule:
                    print('no values for more-than rule, skipping')
                    return False

            if np_band.dtype == np.dtype('b') and "values" in rule:
                counter = np.zeros(255, dtype="int32")
                for j in range(np_band.shape[1]):
                    for i in range(np_band.shape[0]):
                        counter[np_band[i, j]] += 1
                counter = counter / (np_band.shape[0] * np_band.shape[1])

                for v in rule["values"]:
                    frac_sum += counter[v]
            else:
                for j in range(np_band.shape[1]):
                    for i in range(np_band.shape[0]):
                        frac_sum += np_band[i, j]
                frac_sum /= (np_band.shape[0] * np_band.shape[1])

            if "more_than" in rule and frac_sum <= rule["more_than"]:
                return False
            if "less_than" in rule and frac_sum >= rule["less_than"]:
                return False
            return True

        if rule["type"] == "ModelConfidence":
            try:
                vectorDataName = ""
                WMSDataName = ""
                LaserDataName = ""

                for source in self.source_config:
                    if source["type"] == "PostgresImageSource":
                        vectorDataName = source["name"]
                    if source["type"] == "WMSImageSource":
                        WMSDataName = source["name"]
                    if source["type"] == "CompositeImageSource":
                        LaserDataName = source["name"]

                if((vectorDataName == "" and LaserDataName == "")):
                    raise Exception(
                        "Missing config information about both vector data and laser data. Need at least one in order to create a label.")

                ortofoto_path = os.path.join(
                    str(img).replace(vectorDataName, WMSDataName))
                lidar_path = os.path.join(
                    str(img).replace(vectorDataName, LaserDataName))
            except:
                raise Exception("Could not set correct data paths")
            with open(rule["data_generator"], "r") as file:
                data_generator_config = json.load(file)
            if(model_is_confident(data_generator_config, ortofoto_path, lidar_path, img, self.eval_model_checkpoint, self.confidence_threshold)):
                return False
            else:
                return True

        raise NotImplementedError(
            f'Rule type {rule["type"]} is not implemented')

    def evaluate_example(self, example):
        for img_set_name, img_set in self.image_set_dict.items():
            if not img_set.rule:
                continue
            if img_set_name not in example:
                return False
            img = example[img_set_name]

            if not self._evaluate_rule(img, img_set.rule):
                return False

        return True

    def load_example(self, i, j):
        example = {}
        for img_set_name in self.image_set_dict:
            img_set = self.image_set_dict[img_set_name]
            img = Tile(img_set.image_source, i, j, img_set.tile_size)
            example[img_set_name] = img
        else:
            if self.evaluate_example(example):
                if self.eager_load:
                    for v in example.values():
                        if v.array is None:
                            raise ValueError("Tile without data?")
                return example
        return None

    def assemble_data(self, region):
        """Generate cached images for an area, possibly satisfying the test implemented in the callable
         'evaluate' and return list of examples. An 'example' is a dictionary mapping image set names to
         image paths in the cache. The 'evaluate(example)' should return True if the example is good.
         For eager loading of data set eager_load=True"""

        number_of_examples = 0  # Images added to dataset
        skipped_images = 0  # Images looked at but not added due to dataset rule

        # Stop searching for confidence images if dataset not filled after this
        search_limit = 10000

        # If running data teacher - check for infinite search, meaning the threshold is set too strict and no data fits the rule

        region_data = self.tile_grid.generate_ij(region)
        region_data = list(region_data)

        if self.shuffle_data:
            # Have to convert to list in order to shuffle data
            random.shuffle(region_data)

        num_images_in_region = len(region_data)
        dataset_region_size = self.max_size if self.max_size else num_images_in_region

        in_q = out_q = None
        processing_set = set()
        processes = []
        if self.num_processes:
            in_q = mp.Queue()
            out_q = mp.Queue()
            for i in range(self.num_processes):
                proc = mp.Process(
                    None, self, f"DatasetBuilder_{i}", args=(out_q, in_q))
                proc.start()
                processes.append(proc)

        for i, j in region_data:
            print(i, j)

            if(self.max_size and self.max_size <= number_of_examples):
                print(
                    "\n !! Dataset reached its max size, stopped dataset production \n")
                break

            if(self.has_model_confidence_rule and skipped_images > search_limit):
                print(
                    f"\n !! Dataset stopped at {number_of_examples} instances, stopped after {search_limit} instances that was skipped due to confidence rule \n")
                break

            if self.num_processes:
                ij = (i, j)
                out_q.put(ij)
                processing_set.add(ij)
                try:
                    while True:
                        ije = in_q.get_nowait()
                        processing_set.remove((ije[0], ije[1]))
                        example = ije[2]
                        if example:
                            number_of_examples += 1
                            print(
                                f"\nAdded to dataset, total instances: {number_of_examples} of {dataset_region_size}.")
                            yield example
                        else:
                            print("Skipping image, didn't pass dataset rule")
                            skipped_images += 1
                except queue.Empty:
                    continue
            else:
                example = self.load_example(i, j)
                if example:
                    number_of_examples += 1
                    print(
                        f"\nAdded to dataset, total instances: {number_of_examples} of {dataset_region_size}")
                    yield example
                else:
                    print("Skipping image, didn't pass dataset rule")
                    skipped_images += 1

        if out_q:
            out_q.put(None)
            while processing_set:
                # Wait until all requested tiles are processed
                ije = in_q.get()
                processing_set.remove((ije[0], ije[1]))
                example = ije[2]
                if example:
                    number_of_examples += 1
                    print(
                        f"\nAdded to dataset, total instances: {number_of_examples} of {dataset_region_size}")
                    yield example
                else:
                    print("Skipping image, didn't pass dataset rule")
                    skipped_images += 1

    def __call__(self, in_q, out_q):
        """Start a server process for image creation"""

        while True:
            ij = in_q.get()
            if ij is None:
                # Work finished, resend finished message for others to read
                in_q.put(None)
                return
            i = ij[0]
            j = ij[1]
            example = self.load_example(i, j)
            out_q.put((i, j, example))
