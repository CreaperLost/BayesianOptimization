import optuna
import mango
from mango.domain.distribution import loguniform
import optuna
from scipy.stats import loguniform,uniform


def objective(trial):
    print(2**-10, 2**10)
    x = trial.suggest_loguniform('x', 2**-10, 2**10)
    return x

study = optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler())
study.optimize(objective, n_trials=5000)
optuna_values =[trial.values[0] for trial in study.trials]
best_trial = study.best_trial
best_x = best_trial.params['x']
best_value = best_trial.value


import mango
from mango.tuner import Tuner
def objective(args_list):
    return [i['x'] for i in args_list]

param_space = {'x': uniform(0.0001, 2**10)}
experiment = Tuner(
    param_dict=param_space,
    objective=objective,conf_dict={'initial_random':5000,'num_iteration':0})
result = experiment.maximize()
mango_values = [i['x'] for i in result['random_params']]


import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(list1, list2):
    # Plotting parameters


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))



    # Plot histogram for list1
    ax1.hist(list1,bins=100,
             alpha=0.5, label='Mango', color='blue')

    # Plot histogram for list2
    ax2.hist(list2, bins=100,
             alpha=0.5, label='Optuna', color='orange')

    # Set plot title and labels
    plt.title('Distribution of Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()

plot_distribution(mango_values, optuna_values)


def calculate_statistics(list1, list2):
    # Calculate statistics for list1
    list1_min = np.min(list1)
    list1_max = np.max(list1)
    list1_median = np.median(list1)
    list1_mean = np.mean(list1)
    list1_percentiles = np.percentile(list1, [25, 50, 75])

    # Calculate statistics for list2
    list2_min = np.min(list2)
    list2_max = np.max(list2)
    list2_median = np.median(list2)
    list2_mean = np.mean(list2)
    list2_percentiles = np.percentile(list2, [25, 50, 75])

    # Print the statistics
    print("Statistics for List 1:")
    print("Min:", list1_min)
    print("Max:", list1_max)
    print("Median:", list1_median)
    print("Mean:", list1_mean)
    print("25th Percentile:", list1_percentiles[0])
    print("50th Percentile:", list1_percentiles[1])
    print("75th Percentile:", list1_percentiles[2])
    print()

    print("Statistics for List 2:")
    print("Min:", list2_min)
    print("Max:", list2_max)
    print("Median:", list2_median)
    print("Mean:", list2_mean)
    print("25th Percentile:", list2_percentiles[0])
    print("50th Percentile:", list2_percentiles[1])
    print("75th Percentile:", list2_percentiles[2])


calculate_statistics(mango_values, optuna_values)