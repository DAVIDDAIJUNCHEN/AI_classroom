#! /usr/bin/env python3

# modules
import numpy as np
import random
import matplotlib.pyplot as plt

# forward propagation
def forwardNN(X_in, W1, b1, W2, b2, actfunc1, actfunc2):
    """Forward NN with one hidden layer"""
    Y1 = []
    F1 = []
    Y2 = []
    F2 = []
    for i in range(len(X_in)):
        x_in = X_in[i,:]
        y1 = np.matmul(W1, np.transpose(np.matrix(x_in))) + b1
        Y1.append(y1)
        f1 = actfunc1(y1)
        F1.append(f1)
        y2 = np.matmul(W2, f1) + b2
        Y2.append(y2)
        F2.append(actfunc2(y2))

    return Y1, F1, Y2, F2

def sigmoid(x):
    """return sigmoid value of x"""
    return 1.0 / (1 + np.exp(-1.0*x))

def sigmoidVec(y):
    """return sigmoid value of each element"""
    row, col = np.shape(y)
    y_new = np.zeros_like(y, float)

    for i in range(row):
        for j in range(col):
            y_new[i, j] = sigmoid(y[i, j])
    return y_new

# derivative of activation function
def der_actfunc2(x):
    """derivative of actfunc2"""
    return sigmoid(x) * (1 - sigmoid(x))

def der_actfunc1(x):
    """derivative of actfunc2"""
    return sigmoid(x) * (1 - sigmoid(x))

# derivatives of E (over y2)
def der_E(label, y2):
    """derivative of E over y2"""
    return y2 - label


# derivatives of f2 in second layer (over W2, b2, f1)
def der_f2_W2(f1, y2, W2):
    """derivative of f2 over W2"""
    len_y2 = len(y2)
    row_W2, col_W2 = np.shape(W2)
    f2_W2 = []
    for l in range(len_y2):
        der_mat = np.zeros_like(W2, float)
        for i in range(row_W2):
            if i == l:
                for j in range(col_W2):
                    der_mat[i, j] = der_actfunc2(y2[l]) * f1[j]
            else:
                der_mat[i, j] = 0
        f2_W2.append(der_mat)

    return f2_W2

def der_f2_b2(y2, b2):
    """derivative of f2 over b2"""
    len_y2 = len(y2)
    row_b2, col_b2 = np.shape(b2)
    f2_b2 = []
    for l in range(len_y2):
        der_mat = np.zeros_like(b2, float)
        for i in range(row_b2):
            if i == l:
                for j in range(col_b2):
                    der_mat[i, j] = der_actfunc2(y2[l])
            else:
                der_mat[i, j] = 0
        f2_b2.append(der_mat)

    return f2_b2

def der_f2_f1(y2, W2):
    """derivative of f2 over f1"""
    len_y2 = len(y2)
    len_y1 = np.shape(W2)[1]
    der_mat = np.zeros(shape=(len_y2, len_y1))
    for l in range(len_y2):
        for t in range(len_y1):
            der_mat[l, t] = der_actfunc2(y2[l])*W2[l, t]

    return der_mat


# derivatives of f1 in 1st layer (over W1, b1)
def der_f1_W1(x, y1, W1):
    """derivative of f1 over W1"""
    len_y1 = len(y1)
    row_W1, col_W1 = np.shape(W1)
    f1_W1 = []

    for t in range(len_y1):
        der_mat = np.zeros_like(W1, float)
        for i in range(row_W1):
            if i == t:
                for j in range(col_W1):
                    der_mat[i, j] = der_actfunc1(y1[t]) * x[j]
            else:
                der_mat[i, j] = 0
        f1_W1.append(der_mat)

    return f1_W1

def der_f1_b1(y1, b1):
    """derivative of f2 over b2"""
    len_y1 = len(y1)
    row_b1, col_b1 = np.shape(b1)
    f1_b1 = []
    for t in range(len_y1):
        der_mat = np.zeros_like(b1, float)
        for i in range(row_b1):
            if i == t:
                for j in range(col_b1):
                    der_mat[i, j] = der_actfunc1(y1[t])
            else:
                der_mat[i, j] = 0
        f1_b1.append(der_mat)

    return f1_b1


# backward propagation
def backprop(X_in, Label, y1, f1, y2, f2, W2, b2, W1, b1, lr=10e-2):
    """back propagation"""
    # update W2 & b2
    W2_lst = []
    b2_lst = []
    W1_lst = []
    b1_lst = []
    num_sample = len(X_in)
    for i in range(num_sample):
        label_i = np.array(Label[i])
        X_i = np.array(X_in[i, :])
        der_E_W2 = der_E(label_i, y2[i]) * sum(der_f2_W2(f1[i], y2[i], W2))

        W2 = W2 - lr * der_E_W2
        W2_lst.append(W2)

        der_E_b2 = der_E(label_i, y2[i]) * sum(der_f2_b2(y2[i], b2))
        b2 = b2 - lr * der_E_b2
        b2_lst.append(b2)

        # update W1 & b1
        der_E_W1 = np.array(np.matmul(np.transpose(der_f2_f1(y2[i], W2)), der_E(label_i, y2[i]))) * \
                   sum(der_f1_W1(X_i, y1[i], W1))
        W1 = W1 - lr * der_E_W1
        W1_lst.append(W1)

        der_E_b1 = np.array(np.matmul(np.transpose(der_f2_f1(y2[i], W2)), der_E(label_i, y2[i]))) * \
                   sum(der_f1_b1(y1[i], b1))
        b1 = b1 - lr * der_E_b1
        b1_lst.append(b1)

    W1 = sum(W1_lst) / num_sample
    W2 = sum(W2_lst) / num_sample
    b1 = sum(b1_lst) / num_sample
    b2 = sum(b2_lst) / num_sample

    return W1, b1, W2, b2


# main function
if __name__ == '__main__':
    # generate samples
    train_X = np.random.normal(0, 1, [100, 3])
    train_Y = 2*train_X[:, 0] + 5*train_X[:, 1] + 10*train_X[:, 2]
    train_Y = np.array([[y] for y in train_Y])

    W1 = np.array([[0.1, 0.6, 0.7], [0.9, 0.2, 0.5]])
    b1 = np.array([[0.4], [0.5]])
    W2 = np.array([[0.9, 0.1]])
    b2 = np.array([[0.5]])
    disp_iter = 10
    total_iter = 10000

    for iter in range(total_iter):
        y1, f1, y2, f2 = forwardNN(train_X, W1, b1, W2, b2, sigmoidVec, sigmoidVec)
        W1, b1, W2, b2 = backprop(train_X, train_Y, y1, f1, y2, f2, W2, b2, W1, b1)

        if iter % disp_iter == 0:
            pred = np.array([np.array(y2[i])[0].tolist() for i in range(len(y2))])
            print('Loss at %s: ' % iter, sum((pred - train_Y)**2))
