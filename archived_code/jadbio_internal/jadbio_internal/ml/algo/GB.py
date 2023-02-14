from jadbio_internal.ml.algo.dt.SplitFunction import SplitFunction
from jadbio_internal.ml.algo.gb.SamplingMethod import SamplingMethod
from jadbio_internal.ml.algo.gb.TreeSpecifier import TreeSpecifier

class GB:
    learningrate =0
    alpha = 0
    l = 0,
    max_delta_step = 0
    max_depth = 0
    min_child_weight = 0
    subsample = 0
    sampling_method = None
    col_sample_by_model = None
    gamma = 0
    maxmodels = 0
    early_stopping = 0
    specifier_factory = 0

    def __init__(self,
                 learningrate: float,
                 alpha: float,
                 l: float,
                 max_delta_step: float,
                 max_depth: int,
                 min_child_weight: float,
                 subsample: float,
                 sampling_method: SamplingMethod,
                 col_sample_by_model: SplitFunction,
                 gamma: float,
                 maxmodels: float,
                 early_stopping: int,
                 specifier_factory: TreeSpecifier
    ):
        self.learningrate = learningrate
        self.alpha = alpha
        self.l = l,
        self.max_delta_step = max_delta_step
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.sampling_method = sampling_method
        self.col_sample_by_model = col_sample_by_model
        self.gamma = gamma
        self.maxmodels = maxmodels
        self.early_stopping = early_stopping
        self.specifier_factory = specifier_factory

    def to_dict(self):
        return {
            'type': 'GradientBoostingTreeTrainer',
            'learningrate': self.learningrate,
            'alpha': self.alpha,
            'l': self.l,
            'maxDeltaStep': self.max_delta_step,
            'maxDepth': self.max_depth,
            'minChildWeight': self.min_child_weight,
            'subsample': self.subsample,
            'samplingMethod': self.sampling_method.value,
            'colSampleByModel': self.col_sample_by_model.to_dict(),
            'gamma': self.gamma,
            'maxmodels': self.maxmodels,
            'earlyStopping': self.early_stopping,
            'specifierFactory': self.specifier_factory.to_dict()
        }
