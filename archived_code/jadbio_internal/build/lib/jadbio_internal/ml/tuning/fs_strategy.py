def auto():
    return AutoFS()


def static(feature_selectors):
    return Static(feature_selectors)


def ai_supplementary(perform_fs:bool, max_signature_size: int, max_signature_count: int, fs_exclude):
    return AISupplementaryFS(
        perform_fs=perform_fs,
        max_signature_size=max_signature_size,
        max_signature_count=max_signature_count,
        fs_exclude=fs_exclude
    )


class FSStrategy:
    pass


class AutoFS(FSStrategy):
    def to_dict(self):
        return {
            'type': 'auto'
        }


class AISupplementaryFS(FSStrategy):
    perform_fs: bool
    max_signature_size: int
    max_signature_count: int
    fs_exclude = None

    def __init__(self, perform_fs: bool, max_signature_size: int, max_signature_count: int, fs_exclude):
        self.perform_fs = perform_fs
        self.max_signature_size = max_signature_size
        self.max_signature_count = max_signature_count
        self.fs_exclude = fs_exclude

    def to_dict(self):
        return {
            'type': 'extended',
            'fsPreference': self.perform_fs,
            'maxSignatureSize': self.max_signature_size,
            'maxSignatureCount': self.max_signature_count,
            'fsExclude': self.fs_exclude
        }


class Static(FSStrategy):
    feature_selectors = None

    def __init__(self, feature_selectors):
        if feature_selectors is None or len(feature_selectors)==0:
            raise 'invalid strategy'
        self.feature_selectors = feature_selectors

    def to_dict(self):
        return {
            'type': 'static',
            'featureSelectors': [fs.to_dict() for fs in self.feature_selectors]
        }