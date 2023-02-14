from enum import Enum

class SamplingMethod(Enum):
    NO_REPLACEMENT = "NO_REPLACEMENT"
    WITH_REPLACEMENT = "WITH_REPLACEMENT"
    GRADIENT = "GRADIENT"