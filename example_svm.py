from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import numpy as np
from fmin.turbo_1 import Turbo1
from fmin.turbo_m import TurboM
from timeit import timeit
import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
from fmin.Random_Forest_Based_Local_BO import Random_Forest_1
# Load the breast cancer dataset
data = load_breast_cancer()
# Define the KFold cross-validator
cv = KFold(n_splits=5, shuffle=True)

class SVM_benchmark:
    def __init__(self,data,cv,dim=2):
        self.dim = dim
        self.lb = np.array([0.1,0.001])
        self.ub = np.array([10,0.1])
        self.X = data.data
        self.y = data.target
        self.cv = cv
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        C,gamma = x[0],x[1]
        auc_scores = []
        for train_index, test_index in cv.split(self.X):
            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]
            svm = SVC(C=C, kernel='rbf', gamma=gamma)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            auc_scores.append(roc_auc_score(y_test, y_pred))
        return -np.mean(auc_scores)



f = SVM_benchmark(data,cv)


n_iterations = 100
n_initial = 10
"""
Create optimizer.
"""

t1_start = timeit()
turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=n_initial,  # Number of initial bounds from an Latin hypercube design
    max_evals = n_iterations,  # Maximum number of evaluations
    batch_size=10,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo1.optimize()

X = turbo1.X  # Evaluated points
fX = turbo1.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
end_time = timeit()-t1_start
print('Time to eval %d \n',end_time)


"""fig = plt.figure(figsize=(7, 5))
matplotlib.rcParams.update({'font.size': 16})
plt.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
plt.plot(np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
plt.title("SVM Benchmark")
plt.tight_layout()
plt.show()"""

from fmin.bayesian_optimization import bayesian_optimization
robo_start = timeit()
# start Bayesian optimization to minimize the objective function
results = bayesian_optimization(f, f.lb, f.ub, model_type="gp", maximizer="differential_evolution" ,n_init=n_initial,num_iterations=n_iterations)
f_best, x_best =  results["f_opt"] , results["x_opt"]
fX = results['incumbent_values']
print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
end_time = timeit() - robo_start
print('Time to eval %d \n',end_time)


"""fig = plt.figure(figsize=(7, 5))
matplotlib.rcParams.update({'font.size': 16})
plt.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
plt.plot(np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
plt.title("SVM Benchmark")
plt.tight_layout()
plt.show()"""




tM_start = timeit()
turboM = TurboM(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=n_initial,  # Number of initial bounds from an Latin hypercube design
    max_evals = n_iterations,  # Maximum number of evaluations
    batch_size=10,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
    n_trust_regions=5
)

turboM.optimize()

X = turboM.X  # Evaluated points
fX = turboM.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
end_time = timeit() - tM_start
print('Time to eval %d \n',end_time)


"""
Turbo  - 1 
Iterations : 1000
Time: 1.46
Best value found:
        f(x) = -0.931
Observed at:
        x = [0.895 0.001]

Robo 
Iterations 500
Best value found:
        f(x) = -0.931
Observed at:
        x = [2.13e+00 1.00e-03]



Turbo  - M
Best value found:
        f(x) = -0.931
Observed at:
        x = [0.895 0.001]

"""

forest1_start = timeit()
Forest_1 = Random_Forest_1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=n_initial,  # Number of initial bounds from an Latin hypercube design
    max_evals = n_iterations,  # Maximum number of evaluations
    verbose=True,  # Print information from each batch
    )

Forest_1.optimize()

X = Forest_1.X  # Evaluated points
fX = Forest_1.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
end_time = timeit() - forest1_start
print('Time to eval %d \n',end_time)



