'''
Author: Y. SHEN
Data: 2020.9.25
Version : 1.0
Discription: training a linear model with the providied data
'''


from data import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
# shut down all warnings
import warnings
warnings.filterwarnings("ignore")
import math
from sklearn.linear_model import LinearRegression


def draw_f():
    x = y = np.linspace(-20, 20, 200)
    X, Y = np.meshgrid(x, y)

    Z = -5 * np.exp(-X * X / 80 - Y * Y / 20) + 0.007 * X * X + 0.003 * Y * Y

    fig = plt.figure()
    ax = Axes3D(fig)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


def draw_points(hist_x, hist_y, hist_f, draw_step=1,
                show_points=True, show_loss=True, yscale_in_log=True):
    if show_points:
        x = y = np.linspace(-20, 20, 200)
        X, Y = np.meshgrid(x, y)

        Z = -5 * np.exp(-X * X / 80 - Y * Y / 20) + \
            0.007 * X * X + 0.003 * Y * Y

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='cool', alpha=0.4)

        ax.scatter(hist_x[::draw_step], hist_y[::draw_step],
                   hist_f[::draw_step], c='r', marker='o')
        plt.show()

    if show_loss:
        loss = hist_f - np.min(Z)
        plt.plot(np.arange(len(loss)), loss)
        if yscale_in_log:
            pyplot.yscale('log')
        plt.show()


def train_linear_model(input_path='input_data.txt', output_path='output_data.txt', show_info=True):

    # load the data
    input_data = np.loadtxt(input_path)[:7]
    output_data = np.loadtxt(output_path)
    if show_info:
        print('Input training shape:', input_data.shape)
        print('Output training shape:', output_data.shape)

    model = LinearRegression()
    model.fit(input_data.T, output_data.T)

    predict = model.predict(input_data.T)
    res = np.sum((predict - output_data.T)**2) / output_data.shape[1]
    if show_info:
        print('Average L1-residual on training set is ', res)

    return model


def draw_best_path(x_0=-10, y_0=-10, ite=100, initial_step_size=1):
    hist_x0, hist_x1, hist_y0, hist_y1, hist_grad_x, hist_grad_y, hist_f, hist_eta, hist_theta, loss = heavy_ball(
        x0=x_0, y0=y_0, ite=ite, inital_step_size=initial_step_size)
    draw_points(hist_x1, hist_y1, hist_f, draw_step=1)


def test_with_linear_model(model, ite=30, x_0=-10, y_0=-10,
                           stopping_criterion=0.2, initial_step_size=1,
                           flag_plot=True, flag_show_info=True):
    hist_x = []
    hist_y = []
    hist_f = []

    grad_x, grad_y = num_grad_f(x_0, y_0)
    x = x_0 - initial_step_size * grad_x
    y = y_0 - initial_step_size * grad_y
    grad_x, grad_y = num_grad_f(x, y)
    init = np.array([x_0, y_0, x, y, grad_x, grad_y, f(x, y)]).reshape((1, 7))
    result = model.predict(init)
    eta = result[0, 0]
    theta = result[0, 1]

    for i in range(ite):
        x_tmp = x - eta * grad_x + theta * (x - x_0)
        y_tmp = y - eta * grad_y + theta * (y - y_0)
        hist_x.append(x_tmp)
        hist_y.append(y_tmp)
        hist_f.append(f(x_tmp, y_tmp))
        x_0 = x
        y_0 = y
        x = x_tmp
        y = y_tmp
        grad_x, grad_y = num_grad_f(x, y)
        init = np.array(
            [x_0, y_0, x, y, grad_y, grad_x, f(x, y)]).reshape((1, 7))
        result = model.predict(init)
        eta = result[0, 0]
        theta = result[0, 1]
        if (f(x, y) - f_min()) < stopping_criterion:
            break
    if flag_show_info:
        print('total iteration times: ', i)
        print('minimal loss is', min(hist_f) - f_min())
    if flag_plot:
        draw_points(hist_x, hist_y, hist_f)
