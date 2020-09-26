'''
Author: Y. SHEN
Data: 2020.9.25
Version : 1.0
Discription: generating the data to train a mode
             generated data is saved in '.txt' format
'''

import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math


def f(x, y):  # define the target function f
    return (-5 * math.exp(-x * x / 80 - y * y / 20) + 0.007 * x * x +
            0.003 * y * y)  # an example


def f_min():  # define the minimal of f
    return f(0, 0)


# numerical compute the gradient
def num_grad_f(x, y, step_size=1e-5):
    grad_x = (f(x + step_size / 2, y) - f(x - step_size / 2, y)) / step_size
    grad_y = (f(x, y + step_size / 2) - f(x, y - step_size / 2)) / step_size
    return grad_x, grad_y


# search paramters in each iteration
def search_para(x0,
                y0,
                x1,
                y1,
                eta_min=-1,
                eta_max=1,
                theta_min=-1,
                theta_max=1,
                n=1000,
                step_size=1e-5):

    eta_delta = (eta_max - eta_min) / n
    theta_delta = (theta_max - theta_min) / n

    # save the current eta and theta
    eta = eta_min
    theta = theta_min

    # save the best parameters associated with the min(f)
    eta_best = eta
    theta_best = theta

    grad_x, grad_y = num_grad_f(x1, y1, step_size)

    # save the best (x,y) associated with the min(f)
    x = x1 - eta * grad_x + theta * (x1 - x0)
    y = y1 - eta * grad_y + theta * (y1 - y0)

    for i in range(n):
        for j in range(n):
            x_tmp = x1 - eta * grad_x + theta * (x1 - x0)
            y_tmp = y1 - eta * grad_y + theta * (y1 - y0)

            if (f(x_tmp, y_tmp) < f(x, y)):  # update the best parameters
                x = x_tmp
                y = y_tmp
                eta_best = eta
                theta_best = theta

            theta += theta_delta
        eta += eta_delta
        theta = theta_min
    return x, y, grad_x, grad_y, eta_best, theta_best


# generate optimized paramater list associated with the inital data points
def heavy_ball(x0, y0,
               eta_min=-1,
               eta_max=1,
               theta_min=-1,
               theta_max=1,
               step_size=1e-3,
               n=1000,
               ite=100,
               stopping_threshold=1e-3,
               inital_step_size=1):

    # determine when to show the training info
    iteration_show = 5
    show_iteration_info = True

    hist_x0 = []
    hist_x1 = []
    hist_grad_x = []
    hist_y0 = []
    hist_y1 = []
    hist_grad_y = []
    hist_f = []
    hist_eta = []
    hist_theta = []
    loss = []

    grad_x, grad_y = num_grad_f(x0, y0, step_size)
    x1 = x0 - inital_step_size * grad_x
    y1 = y0 - inital_step_size * grad_y

    for i in range(ite):
        x2, y2, grad_x, grad_y, eta, theta = search_para(x0, y0, x1, y1,
                                                         eta_min=eta_min,
                                                         eta_max=eta_max,
                                                         theta_max=theta_max,
                                                         theta_min=theta_min,
                                                         n=n,
                                                         step_size=step_size)

        # save the information
        hist_f.append(f(x2, y2)), loss.append(f(x2, y2) - f(0, 0))
        hist_grad_x.append(grad_x)
        hist_grad_y.append(grad_y)
        hist_eta.append(eta)
        hist_theta.append(theta)
        hist_x1.append(x2)
        hist_y1.append(y2)
        hist_x0.append(x1)
        hist_y0.append(y1)

        x0, x1 = x1, x2
        y0, y1 = y1, y2

        # show searching information
        if i % iteration_show == 0 and show_iteration_info:
            print('In the ', i, '-th step of the search algorithm')

        # convergence criterion
        if abs(f(x2, y2) - f(0, 0)) <= stopping_threshold:
            print('The search algorithm convergence at', i, '-th step')
            break
    return hist_x0, hist_x1, hist_y0, hist_y1, hist_grad_x, hist_grad_y, hist_f, hist_eta, hist_theta, loss


def data_info():
    print('The generated data is characterized as follows:')
    print('0-d: x0')
    print('1-d: y0')
    print('2-d: x1')
    print('3-d: y1')
    print('4-d: grad x')
    print('5-d: grad y')
    print('6-d: f(x,y)')
    print('7-d: iteration times')


# main algorithm parts
if __name__ == '__main__':

    # save the inital points
    list_x = [15, 15, -15, -15, 0, 0, 15, -15]
    list_y = [15, -15, 15, -15, 15, -15, 0, 0]
    num_of_training_point = len(list_x)

    # sace the important parameters
    Para = {'eta_min': -1, 'eta_max': 1, 'theta_min': -1, 'theta_max': 1, 'step_size': 1e-5,
            'n': 2000, 'ite': 200, 'stopping_threshold': 1e-3, 'inital_step_size': 0.5}

    hist_x0 = []
    hist_y0 = []
    hist_x1 = []
    hist_y1 = []
    hist_grad_x = []
    hist_grad_y = []
    hist_f = []
    hist_t = []

    hist_eta = []
    hist_theta = []
    loss = []

    for i in range(num_of_training_point):
        print('currently in the stage : ', i + 1, ' of ',
              num_of_training_point, ' total stage')
        hist_x0_tmp, hist_x1_tmp, hist_y0_tmp, hist_y1_tmp, hist_grad_x_tmp, hist_grad_y_tmp, hist_f_tmp, hist_eta_tmp, hist_theta_tmp, loss_tmp = heavy_ball(list_x[i], list_y[i],
                                                                                                                                                              eta_min=Para[
            'eta_min'],
            eta_max=Para[
            'eta_max'],
            theta_min=Para[
            'theta_min'],
            theta_max=Para[
            'theta_max'],
            step_size=Para[
            'step_size'],
            n=Para['n'],
            ite=Para['ite'],
            stopping_threshold=Para[
            'stopping_threshold'],
            inital_step_size=Para['inital_step_size'])

        hist_x0 += hist_x0_tmp
        hist_y0 += hist_y0_tmp
        hist_x1 += hist_x1_tmp
        hist_y1 += hist_y1_tmp
        hist_grad_x += hist_grad_x_tmp
        hist_grad_y += hist_grad_y_tmp
        hist_f += hist_f_tmp
        hist_t += list(range(len(hist_x0_tmp)))

        hist_eta += hist_eta_tmp
        hist_theta += hist_eta_tmp

    # construct the data into matrix
    input_data = np.array(
        [hist_x0, hist_y0, hist_x1, hist_y1, hist_grad_x, hist_grad_y, hist_f, hist_t])
    # (8,n)-dim, where n is the number of data instances
    print('input_data shape : ', input_data.shape)

    output_data = np.array([hist_eta, hist_theta])
    print('output_data shape : ', output_data.shape)  # (2,n)-dim

    # save the data to txt format
    np.savetxt("input_data.txt", input_data)
    np.savetxt("output_data.txt", output_data)

    print('Successfully saved the data!')
