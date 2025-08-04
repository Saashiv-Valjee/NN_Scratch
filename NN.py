import numpy as np
from activations import sigmoid, sigmoid_derivative, relu, relu_derivative

class NeuralNet: 
    # define implicit network characteristics, output layer size for goal is fixed.
    def __init__(self,input_size,hidden_size):
        # weights and biases for the layers to use. 
        
        # Weights
        # weight matrix connects the layers together and should have dimensions per layer that correspond to: 
        # - Columns : edges per node = Nodes in following layer 
        # - Rows    : Nodes per current layer
        
        # Biases
        # bias matrix is just [nodes per current layer, 1].

        # Starting weights can be random to promote asymmetrical learning (so all nodes don't learn the same thing)
        # making starting weights small helps promote changes in the loss through learning and helps prevents sudden changes in the loss space 
        # apparently some use weight * np.sqrt(2 / input_size), called He Initialization...

        # Layer 1 - Hidden Layer
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        
        # Layer 2 - Output layer
        self.W2 = np.random.randn(1, hidden_size) * 0.01
        self.b2 = np.zeros((1, 1))

        
