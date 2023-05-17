''' 
The Neural Network definitions
'''
from microcnn.value import Value
import random


#################### Neuron, Layer, and Model ############################


class Neuron:
    def __init__(self, in_dim):
        self.w = [Value(random.uniform(-0.1, 0.1), op='w') for _ in range(in_dim)]
        self.b = Value(random.uniform(-0.1, 0.1), op='b')

    def forward(self, X):
        self.X = X
        output = sum([w.data * x.data for w, x in zip(self.w, X)], self.b.data)
        self.z = Value(output, children=self.w+X+[self.b], op='neuron')
        return self.z
    
    def backward(self):
        for w, x in zip(self.w, self.X):
            w.grad = self.z.grad * x.data
            if x.children:
                x.grad = self.z.grad * w.data
        self.b.grad = self.z.grad

    def parameters(self):
        return self.w + [self.b, self.z]


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.neurons = [Neuron(n_inputs) for i in range(n_neurons)]

    def forward(self, X):
        return [neuron.forward(X) for neuron in self.neurons]

    def backward(self):
        for neuron in self.neurons:
            neuron.backward()

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            assert len(X) == layer.n_inputs, "Mismatch in input to the layer size"
            X = layer.forward(X)
        return X
    
    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()


########################## Loss Functions ##############################


class RMSLoss:
    def forward(self, Y_pred, Y_gt):
        self.Y_pred = Y_pred  
        self.Y_diff = [(y_pred.data - y_gt.data) for y_pred, y_gt in zip(Y_pred, Y_gt)]
        self.loss = (sum(y_diff**2 for y_diff in self.Y_diff) / len(Y_pred)) ** 0.5
        self.result = Value(self.loss, children=Y_pred+Y_gt, op='rms')
        
        return self.result
    
    def backward(self):
        self.result.grad = 1.0
        for y_pred, y_diff in zip(self.Y_pred, self.Y_diff):
            y_pred.grad = y_diff / (self.loss * len(self.Y_pred))
    
    
class CrossEntropyLoss:
    def forward(self, Y_pred, Y_gt):
        diff_CE = sum(y_gt * y_pred.log() for y_pred, y_gt in zip(Y_pred, Y_gt))
        loss = -diff_CE
        return loss


######################### Activation Functions #########################


class ReLU:
    def forward(self, layer):
        
        return [neuron.z * max(neuron.z.data, 0) for neuron in layer.neurons]


class LeakyReLU:
    def forward(self, layer):
        return [neuron.z * max(neuron.z.data, 0.01) for neuron in layer.neurons]


class Tanh:
    def forward(self, layer):
        return (neuron.z.tanh() for neuron in layer.neurons)


class Softmax:
    def forward(self, layer_values):
        exp_vals = [val.exp() for val in layer_values]
        exp_sum = sum(exp_vals)
        result = [exp_val / exp_sum for exp_val in exp_vals]
        return result


######################### Optimizers ###################################


class SGD:
    def step(self, vals, lr=1e-3, m=0.9):
        for val in vals:
            val.momentum = m * val.momentum + lr * val.grad
            val.data -= val.momentum
