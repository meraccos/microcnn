import math
from microcnn.value import Value
import random


#################### Neuron, Layer, and Model ############################


class Neuron:
    def __init__(self, input_dim, act_fn):
        self.input_dim = input_dim
        self.act_fn = act_fn
        self.w = [Value(random.uniform(-0.1, 0.1)) for _ in range(input_dim)]
        self.b = Value(random.uniform(-0.1, 0.1))

    def __repr__(self):
        return (
            f"Neuron(w: {[f'{i.data:.3f}' for i in self.w]}"
            + f" b: {self.b.data:.3f}, act: {self.a.data:.3f}"
        )

    def forward(self, X):
        self.z = sum([w * x for w, x in zip(self.w, X)], self.b)
        if self.act_fn:
            self.a = self.act_fn.forward(self.z)
        else:
            self.a = self.z
        return self.a

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_neurons, act_fn):
        self.neurons = [Neuron(n_inputs, act_fn) for i in range(n_neurons)]

    def forward(self, X):
        # print(X)
        return [neuron.forward(X) for neuron in self.neurons]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def backward(self):
        pass


########################## Loss Functions ##############################


class RMSLoss:
    def forward(self, Y_pred, Y_gt):
        diff_squared = sum((y_pred - y_gt)**2 for y_pred, y_gt in zip(Y_pred, Y_gt))
        loss = (diff_squared / len(Y_pred)) ** 0.5
        return loss
    
class CrossEntropyLoss:
    def forward(self, Y_pred, Y_gt):
        diff_CE = sum(y_gt * y_pred.log() for y_pred, y_gt in zip(Y_pred, Y_gt))
        loss = -diff_CE
        return loss


######################### Activation Functions #########################


class ReLU:
    def forward(self, n_out):
        if n_out.data > 0:
            result = n_out + 0
        else:
            result = n_out * 0
        return result


class Tanh:
    def forward(self, n_out):
        result = (math.e ** (n_out) - math.e ** (-n_out)) / (math.e ** (n_out) + math.e ** (-n_out))
        return result


class Softmax:
    def forward(self, layer_values):
        exp_vals = [Value(math.e ** val.data, children=[val]) for val in layer_values]
        exp_sum = sum(exp_vals)
        result = [exp_val / exp_sum for exp_val in exp_vals]
        return result


######################### Optimizers ###################################


class SGD:
    def step(self, vals, lr=1e-3, m=0.9):
        for val in vals:
            val.momentum = m * val.momentum + lr * val.grad
            val.data -= val.momentum
