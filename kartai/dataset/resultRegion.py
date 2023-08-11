from enum import Enum


class ResultRegion(Enum):
    KRISTIANSAND = "kristiansand"
    BALSFJORD = "balsfjord"

    @staticmethod
    def from_str(label):
        if label in ('ksand', 'kristiansand'):
            return ResultRegion.KRISTIANSAND
        if label in ('balsfjord'):
            return ResultRegion.BALSFJORD
        raise NotImplementedError
