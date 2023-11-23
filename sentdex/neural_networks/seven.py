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
E = math.e
# X, y = spiral_data(samples=100, classes=3)
# dense1 = Layer_Dense(2, 3)
# activation1 = Activation_ReLU()
# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()

# dense1.forward(X)
# activation1.forward(dense1.output)

# dense2.forward(activation1.outputs)
# activation2.forward(dense2.output)
# print(activation2.output[:5])

'''
solving for x

e ** x = b
'''

# b = 5.2
# print(np.log(b))
# print(E ** 1.6486586255873816)
softmax_output = [0.7, 0.1, 0.2]
target_ouput = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_ouput[0] + 
         math.log(softmax_output[1])*target_ouput[1]+ 
         math.log(softmax_output[2])*target_ouput[2])
print(loss)