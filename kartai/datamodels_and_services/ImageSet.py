class ImageSet:
    """A image set connecting the image source, the requested tile size and a ruleset"""

    def __init__(self, image_source, tile_size, rule):
        self.image_source = image_source
        self.tile_size = tile_size
        self.rule = rule


def getImageSets(config, image_sources):
    # Used in PredictionArea and therefor needs to be a sepperate method
    image_sets = {}
    for img_set_config in config["ImageSets"]:
        source = image_sources[img_set_config["source"]]
        image_set = ImageSet(source, img_set_config["tile_size"],
                             {
                                 "type": "And",
                                 "rules": img_set_config["rules"]
        } if "rules" in img_set_config else None
        )
        image_sets[img_set_config["name"]] = image_set

    return image_sets
