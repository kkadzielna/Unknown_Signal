import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""from utilities.py"""
def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

"""from utilities.py"""
def view_data_segments(xs, ys, y_est):
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


def linear_x(xs):
    ones = np.ones(xs.shape)
    xs = np.column_stack((ones, xs))
    return xs

def poly_x(xs):
    ones = np.ones(xs.shape)
    xs = np.column_stack((ones, xs, xs**2))
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
    a, b, c = fit_maximum_likelihood_estimate(xs, Y)
    ys = a + b * X + c * X**2
    return ys

def sinus(X, Y):
    xs = sin_x(X)
    a, b = fit_maximum_likelihood_estimate(xs, Y)
    ys = a + b*np.sin(X)
    return ys

def sum_square_error(y_act, y_est):
    return np.sum((y_act - y_est)**2)

def shuffle(data_xs, data_ys):
    shuffled_data = []
    for i in range(len(data_xs)):
        shuffled_data.append((data_xs[i], data_ys[i]))
    shuffled_data = list(shuffled_data)
    np.random.shuffle(shuffled_data)
    shuffled_data = tuple(shuffled_data)
    data_xs = np.array([i[0] for i in shuffled_data])
    data_ys = np.array([i[1] for i in shuffled_data])
    return data_xs, data_ys

def kfold(k, data_xs, data_ys, func_type):
    fold_size = len(data_xs)//k
    cv_error = []
    data_xs, data_ys = shuffle(data_xs, data_ys)
    for i in range(k):
        train_xs = data_xs[fold_size*i:fold_size*(i+1)]
        train_ys = data_ys[fold_size*i:fold_size*(i+1)]
        test_xs = np.concatenate((data_xs[:fold_size*i], data_xs[fold_size*(i+1):]))
        test_ys = np.concatenate((data_ys[:fold_size*i], data_ys[fold_size*(i+1):]))
        if func_type == polynomial:
            train_xs = poly_x(train_xs)
            a, b, c = fit_maximum_likelihood_estimate(train_xs, train_ys)
            yh_test = a + b * test_xs + c * test_xs**2 
        elif func_type == linear:
            train_xs = linear_x(train_xs)
            a, b = fit_maximum_likelihood_estimate(train_xs, train_ys)
            yh_test = a + b * test_xs
        elif func_type == sinus:
            train_xs = sin_x(train_xs)
            a, b = fit_maximum_likelihood_estimate(train_xs, train_ys)
            yh_test = a + b*np.sin(test_xs)
        cv_error.append(sum_square_error(test_ys, yh_test))
    cv_error = np.mean(cv_error)
    return cv_error

def repeat(n, k, data_xs, data_ys, func_type):
    cverror = []
    for i in range(n):
        cverror.append(kfold(k, data_xs, data_ys, func_type))
    return np.mean(cverror)
        
def main():
    if len(sys.argv) >= 1:
        filename = str(sys.argv[1])
    xs, ys = load_points_from_file(filename)

    file_len = len(xs)//20
    y_est = []
    k = 5
    n = 50
    for i in range(file_len):
        xs_temp = xs[20*i:20*(i+1)]
        ys_temp = ys[20*i:20*(i+1)]
        y_est_temp_linear = linear(xs_temp, ys_temp)
        y_est_temp_polynomial = polynomial(xs_temp, ys_temp)
        y_est_temp_sin = sinus(xs_temp, ys_temp)

        cverror_linear = repeat(n, k, xs_temp, ys_temp, linear)
        cverror_polynomial = repeat(n, k, xs_temp, ys_temp, polynomial)
        cverror_sin = repeat(n, k, xs_temp, ys_temp, sinus)
        if cverror_linear <= cverror_polynomial and cverror_linear <= cverror_sin:
            y_est_temp = y_est_temp_linear
            cverror = cverror_linear
        else:
            y_est_temp = y_est_temp_polynomial
            cverror = cverror_polynomial
        """else:
            y_est_temp = y_est_temp_sin
            cverror = cverror_sin"""
        y_est = np.concatenate((y_est, y_est_temp))
    sse  = sum_square_error(ys, y_est)
    print(sse)
    print('\n')

    if len(sys.argv) >= 3:
        if str(sys.argv[2]) == "--plot":
            view_data_segments(xs, ys, y_est)


main()
