from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import numpy as np
from timeit import timeit
import numpy as np
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer



"""Should change the default data and default CV."""
class SVM_benchmark_w_ConfigSpace:
    def __init__(self,data = load_breast_cancer(),cv=KFold(n_splits=5, shuffle=True),dim=2):
        self.dim = dim
        C = Integer('C', bounds=(0.1, 10), default=2)
        gamma = Float('gamma', bounds=(0.001, 0.1), default=1, log=True)

        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters([C, gamma])

        self.X = data.data
        self.y = data.target
        self.cv = cv
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        C,gamma = x['C'],x['gamma']
        auc_scores = []
        for train_index, test_index in self.cv.split(self.X):
            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]
            svm = SVC(C=C, kernel='rbf', gamma=gamma)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            auc_scores.append(roc_auc_score(y_test, y_pred))
        return -np.mean(auc_scores)


