from enum import Enum


class DatasetType(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    ALL = "all"

class ModelType(Enum):
    RTDETR = "RTDETR"
    YOLO = "YOLO"