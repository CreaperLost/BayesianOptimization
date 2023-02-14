from jadbio_internal.ml.tuning import preprocessing_strategy, fs_strategy, model_strategy
from jadbio_internal.ml.tuning.fs_strategy import FSStrategy
from jadbio_internal.ml.tuning.model_strategy import ModelStrategy
from jadbio_internal.ml.tuning.preprocessing_strategy import PreprocessingStrategy


def auto():
    return AutoTuning()


# Perform ai based preprocessing and modeling but custom fs,
# Takes as argument list of feature selectors
def custom_fs(feature_selectors):
    return Guided(
        preprocessing_strategy.auto(),
        fs_strategy.static(feature_selectors),
        model_strategy.auto()
    )


# Perform AI based preprocessing and fs but custom models,
# Takes as argument list of models
def custom_models(models):
    return Guided(
        preprocessing_strategy.auto(),
        fs_strategy.auto(),
        model_strategy.static(models)
    )


def custom(configurations):
    return Static(configurations)


class TuningParams:
    def to_dict(self):
        pass

    @staticmethod
    def auto():
        return AutoTuning()


class AutoTuning(TuningParams):
    def to_dict(self):
        return {
            'type': 'auto'
        }


class Guided(TuningParams):
    pp_strategy: PreprocessingStrategy
    fs_strategy: FSStrategy
    model_strategy: ModelStrategy

    def __init__(self,
                 pp_strategy: PreprocessingStrategy,
                 fs_strategy: FSStrategy,
                 model_strategy: ModelStrategy
                 ):
        self.pp_strategy = pp_strategy
        self.fs_strategy = fs_strategy
        self.model_strategy = model_strategy

    def to_dict(self):
        return {
            'type': 'guided',
            'preprocessingParams': self.pp_strategy.to_dict(),
            'fsTuningParams': self.fs_strategy.to_dict(),
            'modelTuningParams': self.model_strategy.to_dict()
        }


class Static(TuningParams):
    configurations = None

    def __init__(self, configurations):
        if configurations is None or len(configurations) == 0:
            raise 'invalid configurations'
        self.configurations = configurations


    def to_dict(self):
        return {
            'type': 'static',
            'configurations': [c.to_dict() for c in self.configurations]
        }