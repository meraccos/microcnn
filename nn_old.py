''' 
The Neural Network definitions
'''
from microcnn.value import Value
import random
import math

#################### Neuron, Layer, and Model ############################


class Neuron:
    def __init__(self, in_dim):
        self.w = [Value(random.uniform(-0.1, 0.1), op='w') 
                  for _ in range(in_dim)]
        self.b = Value(random.uniform(-0.1, 0.1), op='b')
        self.z = Value(0.0)

    def forward(self, X):
        self.X = X
        output = sum([w.data * x.data for w, x in zip(self.w, X)], self.b.data)
        self.z = Value(output, children=self.w+X+[self.b], op='neuron')
        return self.z
    
    def backward(self):
        for w, x in zip(self.w, self.X):
            w.grad += self.z.grad * x.data
            x.grad += self.z.grad * w.data                  # fix this
        self.b.grad += self.z.grad  

    def parameters(self):
        return self.w + [self.b, self.z]

class Layer:
    def __init__(self, n_inputs, n_neurons, act_fn=None):
        self.n_inputs = n_inputs
        self.act_fn = act_fn() or Identity()
        self.neurons = [Neuron(n_inputs) for i in range(n_neurons)]
        # self.act_out = []                                                    # fix this

    def forward(self, X):
        neuron_out = [neuron.forward(X) for neuron in self.neurons]
        self.act_out = self.act_fn.forward(neuron_out)
        return self.act_out

    def backward(self):
        self.act_fn.backward()
        for neuron in self.neurons:
            neuron.backward()

    def parameters(self):
        return [param for neuron in self.neurons 
                for param in neuron.parameters()]+self.act_out                # fix this


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            assert len(X) == layer.n_inputs, "Size mismatch"
            X = layer.forward(X)
        return X
    
    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def zero_grad(self, batch=False):
        for param in self.parameters():
            if batch:
                if param.op not in ['w', 'b']:
                    param.grad = 0.0
            else:    
                param.grad = 0.0
                

########################## Loss Functions ##############################


class RMSLoss:
    def forward(self, Y_pred, Y_gt):
        assert len(Y_pred) == len(Y_gt), "Size mismatch"
        self.Y_pred = Y_pred  
        self.Y_diff = [(y_pred.data - y_gt.data) for y_pred, y_gt in zip(Y_pred, Y_gt)]
        self.loss = (sum(dy**2 for dy in self.Y_diff) / len(Y_pred)) ** 0.5
        
        return Value(self.loss, children=Y_pred+Y_gt, op='rms_loss', grad=1.0)
    
    def backward(self):
        for y_pred, y_diff in zip(self.Y_pred, self.Y_diff):
            y_pred.grad = 2 * y_diff / (self.loss * len(self.Y_pred))
    
    
class SoftmaxCrossEntropyLoss:
    def forward(self, Y_pred, Y_gt):
        assert len(Y_pred) == len(Y_gt), "Size mismatch"
        self.Y_pred = Y_pred
        self.Y_gt = Y_gt

        # Apply softmax
        exp_vals = [math.exp(val.data) for val in Y_pred]
        exp_sum = sum(exp_vals)
        self.softmax_data = [exp_val / exp_sum for exp_val in exp_vals]

        # Compute cross-entropy loss
        loss = -sum(y_gt.data * math.log(y_sm) 
                    for y_gt, y_sm in zip(Y_gt, self.softmax_data))
        
        return Value(loss, children=Y_pred+Y_gt, op='softmax_cce_loss', grad=1.0)

    def backward(self):
        for y_pred, y_softmax, y_gt in zip(self.Y_pred, self.softmax_data, self.Y_gt):
            y_pred.grad = y_softmax - y_gt.data
            
            
######################### Activation Functions #########################


class BaseActivation:
    def forward(self, neuron_outs):
        self.outs = [Value(self._activation(val.data), children=[val], op=self.op) 
                     for val in neuron_outs]
        return self.outs

    def backward(self):
        for out in self.outs:
            out.children[0].grad = out.grad * self._derivative(out.data)


class Identity(BaseActivation):
    op = 'identity'
    _activation = staticmethod(lambda x: x)
    _derivative = staticmethod(lambda x: 1)


class ReLU(BaseActivation):
    op = 'relu'
    _activation = staticmethod(lambda x: max(x, 0))
    _derivative = staticmethod(lambda x: x > 0)


class LeakyReLU(BaseActivation):
    op = 'l_relu'
    _activation = staticmethod(lambda x: max(x, 0.01 * x))
    _derivative = staticmethod(lambda x: 1 if x > 0 else 0.01)
    
    
class Tanh(BaseActivation):
    op = 'tanh'
    _activation = staticmethod(math.tanh)
    _derivative = staticmethod(lambda x: 1 - x**2)
        

######################### Optimizers ###################################


class SGD:
    def step(self, vals, lr=1e-3, m=0.9):
        for val in vals:
            val.momentum = m * val.momentum + lr * val.grad
            val.data -= val.momentum
