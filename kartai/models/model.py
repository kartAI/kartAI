from enum import Enum
from kartai.models.segmentation_models import get_bottleneck_cross_SPP_model, get_bottleneck_cross_model, get_bottleneck_model, get_csp_model, get_resnet_model, get_unet_model, get_unet_twin_model, gets_csp_cross_SPP_model, gets_csp_cross_model


class Model(Enum):
    UNET = "unet"
    RESNET = "resnet"
    BOTTLENECK = "bottleneck"
    BOTTLENECK_CROSS = "bottleneck_cross"
    BOTTLENECK_CROSS_SPP = "bottleneck_cross_SPP"
    CSP = "CSP"
    CSP_CROSS = "CSP_cross"
    CSP_CROSS_SPP = "CSP_cross_SPP"
    UNET_TWIN = "unet-twin"
    SEGFORMER = "segformer"

    @property
    def has_stacked_input(self):
        return self != self.UNET_TWIN

    @property
    def has_tuple_input(self):
        return self == self.UNET_TWIN

    @staticmethod
    def get_values():
        return [m.value for m in Model]

    @staticmethod
    def from_str(model):
        if model == "unet":
            return Model.UNET
        if model == "resnet":
            return Model.RESNET
        if model == "bottleneck":
            return Model.BOTTLENECK
        if model == "bottleneck_cross":
            return Model.BOTTLENECK_CROSS
        if model == "bottleneck_cross_SPP":
            return Model.BOTTLENECK_CROSS_SPP
        if model == "CSP":
            return Model.CSP
        if model == "CSP_cross":
            return Model.CSP_CROSS
        if model == "CSP_cross_SPP":
            return Model.CSP_CROSS_SPP
        if model == "unet-twin":
            return Model.UNET_TWIN
        if model == "segformer":
            return Model.SEGFORMER
        else:
            raise ValueError(f"Implementation of model {model} does not exist")

    @staticmethod
    def get_model_function(model):
        """Get model function for the given architecture name"""

        if model == Model.UNET:
            return get_unet_model
        if model == Model.RESNET:
            return get_resnet_model
        if model == Model.BOTTLENECK:
            return get_bottleneck_model
        if model == Model.BOTTLENECK_CROSS:
            return get_bottleneck_cross_model
        if model == Model.BOTTLENECK_CROSS_SPP:
            return get_bottleneck_cross_SPP_model
        if model == Model.CSP:
            return get_csp_model
        if model == Model.CSP_CROSS:
            return gets_csp_cross_model
        if model == Model.CSP_CROSS_SPP:
            return gets_csp_cross_SPP_model
        if model == Model.UNET_TWIN:
            return get_unet_twin_model
        if model == Model.SEGFORMER:
            raise ValueError(
                "Segformer is only implemented in stream version")
        else:
            raise ValueError(f"Implementation of model {model} does not exist")
