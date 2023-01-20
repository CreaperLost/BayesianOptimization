from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.datasets import load_breast_cancer,load_digits
from sklearn.model_selection import KFold
import numpy as np
from timeit import timeit
import numpy as np



"""Should change the default data and default CV."""
class SVM_benchmark:
    def __init__(self,data = load_digits(),cv=KFold(n_splits=5, shuffle=True),dim=2):
        self.dim = dim
        self.lb = np.array([0.1,0.001])
        self.ub = np.array([10,0.1])
        self.domain = [[0.1,10],[0.001,0.1]]

        self.X = data.data
        self.y = data.target
        self.cv = cv
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        C,gamma = x[0],x[1]
        auc_scores = []
        for train_index, test_index in self.cv.split(self.X):
            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]
            svm = SVC(C=C, kernel='rbf', gamma=gamma)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            #roc_auc_score(y_test, y_pred)
            auc_scores.append(accuracy_score(y_test, y_pred))
        return -np.mean(auc_scores)


