import numpy as np
import nnfs

# Video 4
# inputs = [[1, 2, 3, 2.5],
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]]
# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
# biases = [2, 3, 0.5]

# # Second layer
# weights2 = [[0.1, -0.14, 0.5],
#            [-0.5, 0.12, -0.33],
#            [-0.44, 0.73, -0.13]]
# biases2 = [-1, 2, -0.5]

# layer1_output = np.dot(inputs, np.array(weights).T) + biases
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
# print(layer2_output)

# np.random.seed(0)


# X = [[1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]

# class Layer_Dense:
#     def __init__(self, n_inputs, n_neurons):
#         self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
#         self.biases = np.zeros((1, n_neurons))
#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases
        
# class Activation_ReLU:
#     def forward(self, inputs):
#         self.outputs = np.maximum(0, inputs)
        
# 
# layer1 = Layer_Dense(4, 5)
# layer2 = Layer_Dense(5, 2)

# layer1.forward(X)
# # print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)


# Video 5
from nnfs.datasets import spiral_data
nnfs.init()


# X = [[1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
# print(layer1.output)
activation1.forward(layer1.output)
print(activation1.outputs)

# np.random.seed(0)
# def create_data(points, classes):
#     X = np.zeros((points*classes, 2))
#     y = np.zeros(points*classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points) # radius
#         t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X, y

# import matplotlib.pyplot as plt

# print("here")
# X, y = create_data(100, 3)
# plt.scatter(X[:,0], X[:,1])
# plt.show()
# plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
# plt.show()