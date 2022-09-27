import numpy as np
import plotly
from matplotlib import pyplot as plt
from more_itertools import powerset

import visualisation
from visualisation import *


class Functor:
    def __init__(self, function, function_name):
        self.__function = function
        self.__function_name = function_name

    def __call__(self, x):
        return self.__function(x)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__function_name

    @property
    def name(self):
        return self.__function_name


class Learning:
    base_functions = [
        Functor(lambda x: np.sin(x), "sin(x)"),
        Functor(lambda x: np.cos(x), "cos(x)"),
        Functor(lambda x: np.log(x + 1e-7), "ln(x+1e-7)"),
        Functor(lambda x: np.exp(x), "exp(x)"),
        Functor(lambda x: np.sqrt(x), "sqrt(x)"),
        Functor(lambda x: x, "x"),
        Functor(lambda x: x ** 2, "x^2"),
        Functor(lambda x: x ** 3, "x^3"),
    ]

    @staticmethod
    def build_design_matrix(functions, X):
        result = np.ones((X.size, len(functions) + 1))
        for i in range(len(X)):
            for j in range(1, len(functions) + 1):
                result[i][j] = functions[j - 1](X[i])
        return result

    @staticmethod
    def learning(design_matrix, t):
        return np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T @ t

    @staticmethod
    def calculate_error(t, W, design_matrix):
        return (1 / 2) * sum((t - (W @ design_matrix.T)) ** 2)

    @staticmethod
    def combinations():
        return list(powerset(Learning.base_functions))[1:93]


def test_learning(X, t, Z):
    base_function = Learning.combinations()
    x_learn, t_learn, x_valid, t_valid, x_test, t_test = train_test_validation_split(X, t)
    data = np.empty((91, 4), dtype="object")
    for i in range(0, 91):
        function = base_function[i]
        F = Learning.build_design_matrix(function, x_learn)
        W = Learning.learning(F, t_learn)
        F_valid = Learning.build_design_matrix(function, x_valid)
        E = Learning.calculate_error(t_valid, W, F_valid)
        test_design = Learning.build_design_matrix(function, x_test)
        test_error = Learning.calculate_error(t_test, W, test_design)
        data[i][0] = function
        data[i][1] = E
        data[i][2] = W
        data[i][3] = test_error
    counter = 1
    while counter < 91:
        for i in range(91 - counter):
            if data[i][1] > data[i + 1][1]:
                data[[i, i + 1]] = data[[i + 1, i]]
        counter += 1
    best_func = []
    best_weight = []
    best_val_err = []
    best_test_err = []
    for i in range(0, 10):
        best_func.append(data[i][0])
        best_weight.append(data[i][2])
        best_test_err.append(data[i][3])
        best_val_err.append(data[i][1])

    for i in range(0, 10):
        print(f"validation error = {data[i][1]}")
        print(f"best weights = {data[i][2]}")
        print(f"best functions = {data[i][0]}")
        print(f"test error= {data[i][3]}\n")

    validation = list(map(int, best_val_err))
    test = list(map(int, best_test_err))

    visualisation = Visualization()
    visualisation.models_error_scatter_plot(validation, test, best_func, title='10 лучших моделей', show=True,
                                            save=True,
                                            name="Homework2",
                                            path2save="C:/plotly/Homework2")


def train_test_validation_split(X, t):
    ind_prm = np.random.permutation(np.arange(N))  # перемешивает данные, это индексы наших массивов
    tr = 0.8
    val = 0.1
    train_ind = ind_prm[:int(tr * N)]  # до 0,8
    valid_ind = ind_prm[int(tr * N):int((val + tr) * N)]  # от 0,8 до 0,9
    test_ind = ind_prm[int((val + tr) * N):]  # от 0,9 до 1
    x_train, x_valid, x_test = X[train_ind], X[valid_ind], X[test_ind]
    t_train, t_valid, t_test = t[train_ind], t[valid_ind], t[test_ind]
    return x_train, t_train, x_valid, t_valid, x_test, t_test


N = 1000
X = np.linspace(0, 1, N)
Z = 20 * np.sin(2 * np.pi * 3 * X) + 100 * np.exp(X)
error = 10 * np.random.randn(N)
t = Z + error

test_learning(X, t, Z)
