import numpy as np

"""
Code from ROBO!
https://github.com/automl/RoBO/blob/master/robo/initial_design/init_grid.py

"""
def init_grid(lower, upper, n_points):
    """
    Returns as initial design a grid where each dimension is split into N intervals
    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bounds of the input space
    upper: np.ndarray (D)
        Upper bounds of the input space
    n_points: int
        The number of points in each dimension
    Returns
    -------
    np.ndarray(N**lower.shape[0], D)
        The initial design data points
    """
    # Number of hyper-parameters 
    n_dims = lower.shape[0]
    #Create a space with 0s
    X = np.zeros([n_points ** n_dims, n_dims])
    # Find the intervals
    intervals = [np.linspace(lower[i], upper[i], n_points) for i in range(n_dims)]
    #Fill in later
    m = np.meshgrid(*intervals)

    #Get the values per dimension.
    for i in range(n_dims):
        X[:, i] = m[i].flatten()

    return X