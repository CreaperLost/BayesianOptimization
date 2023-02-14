class VariableRangeFilterPrep:
    start = 0
    end = 0

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def to_dict(self):
        return {
            'type': 'VariableRangeFilterFactory',
            'from': self.start,
            'to': self.end
        }