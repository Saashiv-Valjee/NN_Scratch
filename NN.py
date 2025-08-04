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

    
    def forward(self,X):
        # forward pass / feed forward for computations 
        # comptuations are the dot product of the initial signals from previous layers + the weights on the edges
        # passed through the layers corresponding activation function 

        # layer 1 - Hidden layer
        self.Z1 = np.dot(self.W1,X) + self.b1
        self.A1 = relu(self.Z1)

        # Layer 2 - Output Layer 
        self.Z2 = np.dot(self.W2,self.A1) + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2
    
    def loss(self,y):
        # need some kind of classification based loss function.
        # binary cross entropy is used most often 
        
        # we can use the current sample size to scale the loss, otherwise loss would scale up with increasing data size
        # y will be the matrix of training data, with the columns being the amount of training data samples 
        m = y.shape[1]

        # if the network predicts exactly 0 or 1 (self.A2), the below will crash
        # we need to circumvent this using some negligable offset
        eps = 1e-15  

        # binary cross entropy
        return -np.mean(y * np.log(self.A2 + eps) + (1 - y) * np.log(1 - self.A2 + eps))


