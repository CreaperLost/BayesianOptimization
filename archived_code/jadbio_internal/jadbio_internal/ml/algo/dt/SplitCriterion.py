from enum import Enum

class SplitCriterion(Enum):
    CMLogRank = "CMLogRankSplittingCriterionFactory"
    Deviance = "DevianceSplittingCriterionFactory"
    MHLogRank = "MHLogRankSplittingCriterionFactory"
    MSES = "MSESplittingCriterionFactory"

    def to_dict(self):
        return {'type': str(self.value) }



# class CMLogRank(SplitCriterion):
#     def to_dict(self):
#         return {'type': 'CMLogRankSplittingCriterionFactory'}
#
# class Deviance(SplitCriterion):
#     def to_dict(self):
#         return {'type': 'DevianceSplittingCriterionFactory'}
#
# class MHLogRank(SplitCriterion):
#     def to_dict(self):
#         return {'type': 'MHLogRankSplittingCriterionFactory'}
#
# class MSES(SplitCriterion):
#     pass