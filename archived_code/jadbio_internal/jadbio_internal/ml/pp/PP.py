from enum import Enum

from jadbio_internal.ml.pp.StandardPP import StandardPP


class PP(Enum):
    Standard = StandardPP()