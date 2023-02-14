
def ai_supplementary(prep_list, order='FIRST'):
    return AISupplementary(prep_list, order)

def auto():
    return Auto()

class AISupplementary:
    prep_list = None
    order = None

    def __init__(self, prep_list, order):
        self.prep_list = prep_list
        self.order = order

    def to_dict(self):
        return {
            'type': 'enhanced',
            'order': self.order,
            'extraPreprocessors': [e.to_dict() for e in self.prep_list]
        }

class Auto:
    def to_dict(self):
        return {
            'type': 'auto'
        }