import numpy as np
from value import Value


########################## Neuron and Layer ##############################


class Neuron:
    def __init__(self, input_dim, act_fn):
        self.input_dim = input_dim
        self.act_fn = act_fn
        self.w = [Value(np.random.randn()) for _ in range(input_dim)]
        self.b = Value(np.random.randn())

    def __repr__(self):
        return (
            f"Neuron(w: {[f'{i.data:.3f}' for i in self.w]}"
            + f" b: {self.b.data:.3f}, act: {self.a.data:.3f}"
        )

    def forward(self, inputs):
        assert len(inputs) == self.input_dim, "Mismatched input dimensions"
        self.z = sum(w * x for w, x in zip(self.w, inputs)) + self.b
        self.a = self.act_fn.forward(self.z)
        return self.a

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_neurons, act_fn):
        self.neurons = [Neuron(n_inputs, act_fn) for i in range(n_neurons)]

    def forward(self, inputs):
        res = []
        for neuron in self.neurons:
            res.append(neuron.forward(inputs))
        return res

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params += neuron.parameters()
        return params


########################## Loss Functions ##############################


class RMSLoss:
    def forward(y_pred, y_gt):
        loss = (y_pred - y_gt) ** 2
        loss = loss**0.5
        return loss


######################### Activation Functions #########################


class ReLU:
    def forward(n_out):
        result = Value(data=max(0, n_out.data), children=[n_out], op="relu")
        return result


class Tanh:
    def forward(n_out):
        result = Value(data=np.tanh(n_out.data), children=[n_out], op="tanh")
        return result


######################### Optimizers ###################################


class SGD:
    def forward(neurons, lr=1e-3, m=0.9):
        for neuron in neurons:
            neuron.momentum = m * neuron.momentum + lr * neuron.grad
            neuron.data -= neuron.momentum
