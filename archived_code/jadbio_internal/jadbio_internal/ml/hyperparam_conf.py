from jadbio_internal.ml.pp.PP import PP


class HyperparamConf:
    pp = None
    fs = None
    model = None

    def __init__(self, pp, fs, model):
        self.pp = pp
        self.fs = fs
        self.model = model

    @staticmethod
    def with_default_pp(fs, model):
        return HyperparamConf(PP.Standard, fs, model)

    def to_dict(self):
        return {'preprocessor': self.pp.value.to_dict(), 'fs': self.fs.to_dict(), 'model': self.model.to_dict()}
