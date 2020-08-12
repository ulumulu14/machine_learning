import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(100, 3)
np.random.seed(0)
''''
class Neuron:
    def __init__(self, neighbors, next = None, previous = None, weights = [], bias = None):
        self.neighbors = neighbors
        self.next = next
        self.previous = previous
        self.weights = weights
        self.bias = bias

    def next(self):
        return self.next

    def previous(self):
        return self.previous

    def decision_function(self, input, weights, bias):
        output = np.dot(input, weights) + bias
        return output
'''
#layers.py

class DenseLayer:
    def __init__(self, n_neurons: int, decision_function: str):
        self.n_neurons = n_neurons
        self.inputs = None
        self.weights = None
        self.biases = np.zeros((1, n_neurons))
        self.decision_function = decision_function

    def forward(self):
        if self.decision_function == "linear":
            return self.linear()
        elif self.decision_function == "relu":
            return self.relu()
        elif self.decision_function == "softmax":
            return self.softmax()

    def read_input(self, inputs):
        self.inputs = inputs
        self.weights = 0.1 * np.random.randn(len(inputs[0]), self.n_neurons)

    def linear(self):
        return np.dot(self.inputs, self.weights) + self.biases

    def relu(self):
        return np.maximum(0, self.inputs)

    def softmax(self):
        return np.exp(self.inputs) / np.sum(np.exp(self.inputs))

    def structure(self):
        structure = "Neurons: {}".format(self.n_neurons) + "\n" + "Decision function: " + self.decision_function + "\n"
        return structure

#networks.py
class NeuralNetwork:
    def __init__(self):
        self.data = None
        self.layers = []

    def add_layer(self, n_neurons, decision_function):
        self.layers.append(DenseLayer(n_neurons=n_neurons, decision_function=decision_function))

    def train(self, data):
        self.data = data
        inputs = self.data

        for layer in self.layers:
            layer.read_input(inputs)
            inputs = layer.forward()

        outputs = inputs

        return outputs

    def backpropagation(self):
        pass

    def structure(self):
        for i, layer in enumerate(self.layers):
            print("Layer {}".format(i + 1))
            print(layer.structure())

    def gradient_descent(self):
        pass

    def stochastic_gradient_descent(self):
        pass

    def adam(self):
        pass


#X = np.array([[1, 2, 3, 2.5],
 #             [2.0, 5.0, -1.0, 2.0],
#              [-1.5, 2.7, 3.3, -0.8]])
'''
print(X)
layer1 = DenseLayer(4, "linear")
layer1.read_input(X)
output1 = layer1.forward()
print(output1)

layer2 = DenseLayer(4, "relu")
layer2.read_input(output1)
output2 = layer1.forward()
print(output2)
'''

nn = NeuralNetwork()
nn.add_layer(5, "linear")
nn.add_layer(5, "relu")
nn.add_layer(2, "softmax")

nn.structure()

output = nn.train(X)
print(output)
