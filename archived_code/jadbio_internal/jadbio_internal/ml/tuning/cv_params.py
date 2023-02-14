def auto():
    return CVParams(None, None, None, None)


def custom_repeats(repeats: int):
    return CVParams(None, repeats, None, None)


class CVParams:
    metric_to_optimize = None
    repeats = None
    is_time_series = None
    group_factor = None

    def __init__(self, metric, repeats, is_time_series, group_factor):
        self.metric_to_optimize = metric
        self.repeats = repeats
        self.is_time_series = is_time_series
        self.group_factor = group_factor

    def to_dict(self):
        return {
            'metric2Optimize': self.metric_to_optimize,
            'repeats': self.repeats,
            'isTimeSeries': self.is_time_series,
            'groupFactor': self.group_factor
        }
