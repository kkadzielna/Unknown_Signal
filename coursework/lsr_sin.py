import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""Read in a file containing a number of different line segments (each made up of 20 points)
    Determine the function type of line segment, 
        linear
        polynomial with a fixed order that you must determine
        unknown nonlinear function that you must determine
    Use maximum-likelihood/least squares regression to fit the function
    Produce the total reconstruction error 
    Produce a figure showing the reconstructed line from the points if an optional argument is given. 
  """

"""from utilities.py"""
def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

"""from utilities.py"""
def view_data_segments(xs, ys, y_est):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])

    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.plot(xs, y_est, 'r-', lw = 1 )
    plt.show()

def fit_maximum_likelihood_estimate(xs, ys):
    result = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(ys)
    return result

#refactor idea: linear, polynomial take in xs for max likelihood est and an X for te exquations
def linear_x(xs):
    ones = np.ones(xs.shape)
    xs = np.column_stack((ones, xs))
    return xs

def poly_x(xs):
    ones = np.ones(xs.shape)
    xs = np.column_stack((ones, xs, xs**2, xs**3))
    return xs

def sin_x(xs):
    ones = np.ones(xs.shape)
    xs = np.column_stack((ones, np.sin(xs)))
    return xs

def linear(X, Y):
    xs = linear_x(X)
    a, b = fit_maximum_likelihood_estimate(xs, Y)
    ys = a + b * X
    return ys

def polynomial(X, Y):
    xs = poly_x(X)
    a, b, c, d = fit_maximum_likelihood_estimate(xs, Y)
    ys = a + b * X + c * X**2 + d * X**3
    return ys

#try exponential or logarithmic for the unknown func
def sinus(X, Y):
    xs = sin_x(X)
    a, b = fit_maximum_likelihood_estimate(xs, Y)
    ys = a * b*np.sin(X)
    return ys

def sum_square_error(y_act, y_est):
    return np.sum((y_act - y_est)**2)

def kfold(k, data_xs, data_ys, func_type):
    fold_size = len(data_xs)//k
    cv_error = []
    for i in range(k):
        train_xs = data_xs[fold_size*i:fold_size*(i+1)]
        train_ys = data_ys[fold_size*i:fold_size*(i+1)]
        test_xs = np.concatenate((data_xs[:fold_size*i], data_xs[fold_size*(i+1):]))
        test_ys = np.concatenate((data_ys[:fold_size*i], data_ys[fold_size*(i+1):]))
        #may be possible to reduce with the new loinear and polynomial function definitions
        if func_type == polynomial:
            train_xs = poly_x(train_xs)
            a, b, c, d = fit_maximum_likelihood_estimate(train_xs, train_ys)
            yh_test = a + b * test_xs + c * test_xs**2 + d * test_xs**3
        elif func_type == linear:
            train_xs = linear_x(train_xs)
            a, b = fit_maximum_likelihood_estimate(train_xs, train_ys)
            yh_test = a + b * test_xs
        elif func_type == sinus:
            train_xs = sin_x(train_xs)
            a, b = fit_maximum_likelihood_estimate(train_xs, train_ys)
            yh_test = a * b*np.sin(test_xs)
            
        #I'd prefer to use the linear, poly etc functions here, but 
        #fit max is supposed to use train_xs and yh_test is supposed to be on test_xs
        #and i can't split it with what the funcs look like now
        cv_error.append(sum_square_error(test_ys, yh_test))
    return cv_error
"""I still need to:
    make sure the fit is decent
    doesn't pick correctly between linear/polynomial, favours linear. Maybe better cross validation? 
    find out how to utilise cross validation
    make it fit an unknown function - probably not exponential
    make it switch between the different function types
    testing
    report"""
def main():
    if len(sys.argv) >= 1:
        filename = str(sys.argv[1])
    xs, ys = load_points_from_file(filename)

    file_len = len(xs)//20
    y_est = []
    for i in range(file_len):
        xs_temp = xs[20*i:20*(i+1)]
        ys_temp = ys[20*i:20*(i+1)]
        y_est_temp_linear = linear(xs_temp, ys_temp)
        y_est_temp_polynomial = polynomial(xs_temp, ys_temp)
        y_est_temp_sin = sinus(xs_temp, ys_temp)
        k = 2
        cverror_linear = kfold(k, xs_temp, ys_temp, linear)
        cverror_polynomial = kfold(k, xs_temp, ys_temp, polynomial) #pick correct k
        cverror_sin = kfold(k, xs_temp, ys_temp, sinus)
        if cverror_linear <= cverror_polynomial and cverror_linear <= cverror_sin:
            print("linear")
            y_est_temp = y_est_temp_linear
            cverror = cverror_linear
        elif cverror_polynomial <= cverror_sin:
            print("poly")
            y_est_temp = y_est_temp_polynomial
            cverror = cverror_polynomial
        else:
            print("sin")
            y_est_temp = y_est_temp_sin
            cverror = cverror_sin
        y_est = np.concatenate((y_est, y_est_temp))
    #print(y_est)
    sse  = sum_square_error(ys, y_est)
    print(sse)

    if len(sys.argv) >= 3:
        if str(sys.argv[2]) == "--plot":
            view_data_segments(xs, ys, y_est)


main()
