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
        # binary cross entropy is used most often due to it being "convex" (less chance to fall into local minima?)
        # but we will use 
        # this function will need the classification of the model and the truth labels y
        
        # we can use the current sample size to scale the loss, otherwise loss would scale up with increasing data size
        # y will be the matrix of training labels, with the columns being the amount of training data samples 
        m = y.shape[1]

        # if the network predicts exactly 0 or 1 (self.A2), the below will crash
        # we need to circumvent this using some negligable offset
        eps = 1e-15  

        # binary cross entropy
        return -np.mean(y * np.log(self.A2 + eps) + (1 - y) * np.log(1 - self.A2 + eps))


    def back_prop(self, X, y, learning_rate = 0.01): 
        # back propogation is essentially just taking the loss function, and continually using the chain rule with partial derivatives of the activation functions and their inputs
        # until the very beginning weights can be related directly to its loss in the loss space

        # by doing this, we can change the weights and check if they cause the loss function gradient to increase or decrease
        # a decreasing loss function means we are improving the learning and we should continue changing the weights in the direction we are currently 

        # this will be explained in depth elsewhere in the repo

        # learning rate corresponds to the amount by which initial weights are altered
        
        # X is the training data, of dimensions [features,samples]
        m = X.shape[1]

        # === Gradients for output layer ===
        # dL/dW2 = dL/dA2 · dA2/dZ2 · dZ2/dW2  = (A2 - y) · A1^T     (from chain rule + vectorization)
        dL_dW2 = (1/m) * np.dot(self.A2 - y, self.A1.T)
        # dL/db2 = dL/dZ2 · dZ2/db2 = sum over samples of dZ2
        dL_db2 = (1/m) * np.sum(self.A2 - y, axis=1, keepdims=True)


        # === Gradients for hidden layer ===
        # dL/dZ2 = dL/dA2 · dA2/dZ2 = A2 - y
        dL_dZ2 = self.A2 - y
        # dL/dA1 = dL/dZ2 · dZ2/dA1 = W2^T · dZ2
        dL_dA1 = np.dot(self.W2.T, dL_dZ2)
        # dL/dZ1 = dL/dA1 · dA1/dZ1 = dA1 * ReLU'(Z1)
        dL_dZ1 = dL_dA1 * relu_derivative(self.Z1)
        # dL/dW1 = dL/dZ1 · dZ1/dW1 = dZ1 · X^T
        dL_dW1 = (1/m) * np.dot(dL_dZ1, X.T)
        # dL/db1 = dL/dZ1 · dZ1/db1 = sum over samples of dZ1
        dL_db1 = (1/m) * np.sum(dL_dZ1, axis=1, keepdims=True)


        # Gradient descent update
        # change the weights based on the change in the gradient wrt the weights
        # if the gradient is positive, move the weight one direction +/-, if it's negative go the other way.
        self.W1 = self.W1 - (learning_rate * dL_dW1)
        self.b1 = self.b1 - (learning_rate * dL_db1)
        self.W2 = self.W2 - (learning_rate * dL_dW2)
        self.b2 = self.b2 - (learning_rate * dL_db2)

# predict all elements using an initial set of weights and bias
# 