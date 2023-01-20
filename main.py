import torch
import math
import matplotlib
import matplotlib.pyplot as plt
from fmin.Random_Forest_Based_Local_BO import Random_Forest_1
import numpy as np
from benchmarks.Levy import Levy_benchmark
from fmin.turbo_1 import Turbo1
from fmin.turbo_m import TurboM
from benchmarks.SVM import SVM_benchmark
from fmin.Robo_BO import Robo
from timeit import timeit
#Define objective functions
from fmin.Robo_Based_Rf import Random_Forest_Robo
from benchmarks.SVM_ConfigSpace import SVM_benchmark_w_ConfigSpace
from dragonfly import minimise_function
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

f_Levy = Levy_benchmark(10)
f_SVM = SVM_benchmark()

#f_SVM_Config_Space  = SVM_benchmark_w_ConfigSpace()
#f_Levy,
Objective_F = [f_SVM]
#Objective_F= [f_SVM_Config_Space]


#Define Optimizers
n_init = 3
n_eval = 50
n_trust_regions = 5

"""
Create optimizer.
"""
#,,, , ,
optimizers = [('Robo',Robo),('Robo_RF_1',Random_Forest_Robo),('Dragonfly',minimise_function)] #('Forest_1',Random_Forest_1),('Turbo_1',Turbo1),('Turbo_M',TurboM)

for f in Objective_F:
    for opt_name,opt_class in optimizers:
        if 'Turbo_1' == opt_name :
            BO_class = opt_class(f=f,  # Handle to objective function 
                      lb=f.lb,  # Numpy array specifying lower bounds
                      ub=f.ub,  # Numpy array specifying upper bounds
                      n_init=n_init,  # Number of initial bounds from an Latin hypercube design
                      max_evals = n_eval,  # Maximum number of evaluations
                      batch_size=10,  # How large batch size TuRBO uses
                      verbose=True,  # Print information from each batch
                      use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                      max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                      n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                      min_cuda=1024,  # Run on the CPU for small datasets
                      device="cpu",  # "cpu" or "cuda"
                      dtype="float64",  # float64 or float32
                    )
        elif 'Turbo_M' == opt_name:
            BO_class = opt_class(
                    f=f,  # Handle to objective function
                    lb=f.lb,  # Numpy array specifying lower bounds
                    ub=f.ub,  # Numpy array specifying upper bounds
                    n_init=n_init,  # Number of initial bounds from an Symmetric Latin hypercube design
                    max_evals=n_eval,  # Maximum number of evaluations
                    n_trust_regions=n_trust_regions,  # Number of trust regions
                    batch_size=10,  # How large batch size TuRBO uses
                    verbose=True,  # Print information from each batch
                    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                    min_cuda=1024,  # Run on the CPU for small datasets
                    device="cpu",  # "cpu" or "cuda"
                    dtype="float64",  # float64 or float32
                )
        elif 'Robo' == opt_name:
            BO_class = opt_class(f, f.lb, f.ub,
                 'ei', 'gp', 'random',
                 initial_points=n_init,
                 output_path=None,
                 train_interval=1,
                 n_restarts=1,
                 rng=None,
                 num_iterations = n_eval)
        elif 'Forest_1' == opt_name:
            BO_class = opt_class(
                f=f,  # Handle to objective function
                lb=f.lb,  # Numpy array specifying lower bounds
                ub=f.ub,  # Numpy array specifying upper bounds
                n_init=n_init,  # Number of initial bounds from an Latin hypercube design
                max_evals = n_eval,  # Maximum number of evaluations
                verbose=True,  # Print information from each batch
            )
        elif 'Robo_RF_1' == opt_name:
            BO_class = opt_class(
                f,  # Handle to objective function
                f.lb,  # Numpy array specifying lower bounds
                f.ub,  # Numpy array specifying upper bounds
                initial_points=n_init,  # Number of initial bounds from an Latin hypercube design
                num_iterations = n_eval,  # Maximum number of evaluations
                output_path=None,
                train_interval=1,
                n_restarts=1,
                rng=None,  # Print information from each batch
            )
        elif 'Dragonfly'  == opt_name:
            BO_class = None
            f_best,x_best,_ =minimise_function(f,f.domain,opt_method = 'bo',max_capital = n_eval)


        """ 
        Call Optimizer
        """
        start = timeit()
        if BO_class != None:
            BO_class.optimize()
            X = BO_class.X  # Evaluated points
            fX = BO_class.fX  # Observed values
            ind_best = np.argmin(fX)
            f_best, x_best = fX[ind_best], X[ind_best, :]

        
        print("Current Optimizer is %s \n\t " % opt_name)
        print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
        print('Time to eval %d \n',timeit() - start)
        fig = plt.figure(figsize=(7, 5))
        matplotlib.rcParams.update({'font.size': 16})
        plt.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
        plt.plot(np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
        plt.xlim([0, len(fX)])
        plt.title("Progression for " + opt_name)
        plt.tight_layout()
        plt.show()
