from enum import Enum

class TreeSpecifier(Enum):
    BINARY= 'BinaryGDTreeSpecifierFactory'
    MULTINOMIAL = 'MultinomGDTreeSpecifierFactory'
    REGRESSION = 'RegressionGDTreeSpecifierFactory'
    SURVIVAL = 'SurvivalGDTreeSpecifierFactory'

    def to_dict(self):
        return {'type': str(self.value) }
