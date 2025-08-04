import numpy as np

def sigmoid(z):
    # good for binary classification
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    # the derivative for gradient descent
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    # good for hidden layers due to easy computation and not scaling inputs to high values
    return np.maximum(0, z)

def relu_derivative(z):
    # derivative is 1 if x > 0, which is set to 1 otherwise 0
    # This is good for efficient gradient descent, which will just stop for non contributing neurons (with a 0 from relu)
    return (z > 0).astype(float)

