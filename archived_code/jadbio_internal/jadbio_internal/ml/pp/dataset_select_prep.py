class DatasetSelectPrep:
    dataset_id = 0

    def __init__(self, dataset_id: int):
        self.dataset_id = dataset_id

    def to_dict(self):
        return {
            'type': 'DatasetSelectPreprocessorFactory',
            'datasetId': self.dataset_id,
        }