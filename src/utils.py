# this file is for outsourcing python functions, to keep our notebooks clean
import numpy as np


def sliced_statistics(y_true, y_pred, number_of_slices):
    '''
    Sort the two arrays y_pred and y_true in y_pred after y_true. Cut
    the sorted arrays into n slices. Return the std and mean of the
    slices.
    y_true: numpy array with shape (x,) with x element of the natural numbers
    y_pred: numpy array with the same shape as y_pred
    n: len of the resulting arrays (int and lower than len(y_true))
    returns: arrays with values of value, mu, sigma

    '''
    # ensure(len(y_true.shape) == 2 and y_true.shape[1] == 1).is_(True)
    # ensure(y_true.shape == y_pred.shape).is_(True)
    # ensure(n).is_an(int)
    # ensure(n < len(y_true)).is_(True)

    # find the sorted order of y_true
    sliceable_length = len(y_true) - (len(y_true) % number_of_slices)
    print(sliceable_length)
    sorted_order = y_true[:sliceable_length].argsort()
    length_of_slice = int(sliceable_length / number_of_slices)
    print(length_of_slice)
    # calculate results
    y_true = y_true[sorted_order].reshape(number_of_slices, length_of_slice)
    y_pred = y_pred[sorted_order].reshape(number_of_slices, length_of_slice)
    value = np.mean(y_true, axis=1)
    mu = np.mean(y_pred, axis=1)
    sigma = np.sqrt(np.mean(np.square(y_true-y_pred), axis=1))
    return value, mu, sigma

def measure_val(X, Y, n):
    '''
    Calculate mean and std of X following Y
    Y: numpy array with shape (,1)
    X: numpy array with the same shape as X
    n: len of each results (int and lower than len(Y))
    returns: value, mu, sigma
    '''
    
    # find the sorted order of Y
    len_by_n = len(Y) - (len(Y) % n)
    perm = Y[:len_by_n, 0].argsort()
    # calculate results
    y = np.mean(np.transpose(Y)[0][perm].reshape(len_by_n/n, n), axis=1)
    mu = np.mean(np.transpose(X)[0][perm].reshape(len_by_n/n, n), axis=1)
    sigma = np.std(np.transpose(X)[0][perm].reshape(len_by_n/n, n), axis=1)
    return y, mu, sigma
