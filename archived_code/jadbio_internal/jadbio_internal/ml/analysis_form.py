from jadbio_internal.ml.tuning import tuning_params, cv_params
from jadbio_internal.ml.tuning.cv_params import CVParams
from jadbio_internal.ml.tuning.tuning_params import TuningParams


def plots_from_analysis_type(type):
    if type == 'CLASSIFICATION':
        return ['Roc', 'Ice', 'Probabilities', 'Pca', 'UMAP']
    return []


def classification(target: str, tuning: str, concurrency: int):
    """

    :param target:
    :param tuning: 'QUICK' | 'NORMAL' | 'EXTENSIVE'
    :param concurrency: 1-24
    :return:
    """
    return AnalysisForm(
        target, target, 'CLASSIFICATION', 'STANDARD', tuning, concurrency
    )

def regression(target: str, tuning: str, concurrency: int):
    """

    :param target:
    :param tuning: 'QUICK' | 'NORMAL' | 'EXTENSIVE'
    :param concurrency: 1-24
    :return:
    """
    return AnalysisForm(
        target, target, 'REGRESSION', 'STANDARD', tuning, concurrency
    )


def testing(target: str, tuning: str, concurrency: int, tuning_params):
    return AnalysisForm(title=target, target=target, analysis_type=None, dataset_type='STANDARD', tuning_effort=tuning, concurrency=concurrency, tuning=tuning_params)

class AnalysisForm:
    target = None
    analysis_type = None  # STANDARD or DISTANCE
    dataset_type = None
    tuning_args: TuningParams = None
    cv_args: CVParams = None
    tuning = None
    concurrency = 1
    timeout = 1000000000
    title = None
    plots = []

    def __init__(
            self,
            title,
            target,
            analysis_type: str,
            dataset_type: str,
            tuning_effort,
            concurrency,
            cv_args=cv_params.auto(),
            tuning=tuning_params.auto(),
            plots=None
    ):
        self.title = title
        self.target = target
        self.analysis_type = analysis_type
        self.dataset_type = dataset_type
        self.tuning = tuning_effort
        self.concurrency = concurrency
        self.plots = plots_from_analysis_type(analysis_type) if plots is None else []
        self.tuning_args = tuning
        self.cv_args = cv_args

    def with_cv_params(self, cv_conf: cv_params):
        return AnalysisForm(
            self.title,
            self.target,
            self.analysis_type,
            self.dataset_type,
            self.tuning,
            self.concurrency,
            cv_conf,
            self.tuning_args
        )

    def with_tuning_strategy(self, tuning_strategy):
        return AnalysisForm(
            self.title,
            self.target,
            self.analysis_type,
            self.dataset_type,
            self.tuning,
            self.concurrency,
            self.cv_args,
            tuning_strategy
        )

    def with_title(self, title):
        return AnalysisForm(
            title,
            self.target,
            self.analysis_type,
            self.dataset_type,
            self.tuning,
            self.concurrency,
            self.cv_args,
            self.tuning_args
        )

    def with_distance_dataset(self):
        return AnalysisForm(
            self.title,
            self.target,
            self.analysis_type,
            'DISTANCE',
            self.tuning,
            self.concurrency,
            self.cv_args,
            self.tuning_args
        )

    def to_dict(self):
        return {
            'title': self.title,
            'target': self.target,
            'analysisType': self.analysis_type,
            'datasetType': self.dataset_type,
            'tuningEffort': self.tuning,
            'coreCount': self.concurrency,
            'plots': self.plots,
            'timeout': self.timeout,
            'cvPreferences': self.cv_args.to_dict(),
            'tuningParams': self.tuning_args.to_dict() if self.tuning_args is not None else None
        }
