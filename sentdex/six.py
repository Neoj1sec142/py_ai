import numpy as np
import nnfs
import math
from nnfs.datasets import spiral_data
nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.outputs)
activation2.forward(dense2.output)
print(activation2.output[:5])


# nnfs.init()
# layer_outputs = [[4.8, 1.21, 2.385],
#                  [8.9, -1.81, 0.2],
#                  [1.41, 1.051, 0.026]]
# # E = math.e

# exp_values = np.exp(layer_outputs)

# # print(np.sum(layer_outputs, axis=1, keepdims=True))

# norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
 
# print(norm_values)