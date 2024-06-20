import numpy as np

class Network(object):
    #creates a network with number of neurons in each respective layer
    # described in a list sizes
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # all weights and biases are randomly initialised from 0 to 1
        # skips first layer weights and biases, treating it as input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        # for each layer, we take the weight and bias matrices
        for b, w in zip(self.biases, self.weights):
            # compute the output matrix of each layer
            # when first called, a is the input vector
            # a then stores the activation vector of the first layer after the first iteration
            # we update a is then used as input to calculate the new activation vector from next layer
            a = sigmoid(np.dot(w, a)+b)
            # finally a is returned at the output layer
        return a
    

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

net = Network([2,3,1])
print(net.weights[0])
