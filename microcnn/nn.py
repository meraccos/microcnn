''' 
The Neural Network definitions
'''
from microcnn.value import Value
import random
import math

#################### Neuron, Layer, and Model ############################


class Neuron:
    def __init__(self, in_dim, act_fn=None):
        self.w = [Value(random.uniform(-0.1, 0.1), op='w') 
                  for _ in range(in_dim)]
        self.b = Value(random.uniform(-0.1, 0.1), op='b')
        self.act_fn = act_fn() or Identity()

    def forward(self, X):
        self.X = X
        output = sum([w.data * x.data for w, x in zip(self.w, X)], self.b.data)
        self.z = Value(output, children=self.w+X+[self.b], op='neuron')
        self.act_out = self.act_fn.forward(self.z)
        return self.act_out
    
    def backward(self):
        self.act_fn.backward()
        for w, x in zip(self.w, self.X):
            w.grad += self.z.grad * x.data
            x.grad += self.z.grad * w.data
        self.b.grad += self.z.grad  

    def parameters(self):
        return self.w + [self.b, self.z, self.act_out]
    
    # def parameters(self):
    #     return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_neurons, act_fn=None):
        self.n_inputs = n_inputs
        self.neurons = [Neuron(n_inputs, act_fn) for i in range(n_neurons)]

    def forward(self, X):
        return [neuron.forward(X) for neuron in self.neurons]

    def backward(self):
        for neuron in self.neurons:
            neuron.backward()

    def parameters(self):
        return [param for neuron in self.neurons 
                for param in neuron.parameters()]


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
    def _apply_softmax(self, Y_pred):
        exp_vals = [math.exp(val.data) for val in Y_pred]
        exp_sum = sum(exp_vals)
        return [exp_val / exp_sum for exp_val in exp_vals]
    
    def forward(self, Y_pred, Y_gt):
        assert len(Y_pred) == len(Y_gt), "Size mismatch"
        self.Y_pred = Y_pred
        self.Y_gt = Y_gt
        self.softmax_data = self._apply_softmax(Y_pred)

        # Compute cross-entropy loss
        loss = -sum(y_gt.data * math.log(y_sm) 
                    for y_gt, y_sm in zip(Y_gt, self.softmax_data))
        
        return Value(loss, children=Y_pred+Y_gt, op='softmax_cce_loss', grad=1.0)

    def backward(self):
        for y_pred, y_softmax, y_gt in zip(self.Y_pred, self.softmax_data, self.Y_gt):
            y_pred.grad = y_softmax - y_gt.data
            
            
######################### Activation Functions #########################


class BaseActivation:
    def forward(self, output):
        self.out = Value(self._activation(output.data), children=[output], op=self.op) 
        return self.out

    def backward(self):
        self.out.children[0].grad = self.out.grad * self._derivative(self.out.data)


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
    def __init__(self, lr=0.01, m=0.9):
        self.lr = lr
        self.m = m
    def step(self, vals):
        for val in vals:
            val.momentum = self.m * val.momentum + self.lr * val.grad
            val.data -= val.momentum


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = t

    def step(self, vals):
        self.t += 1
        for val in vals:
            val.momentum = self.beta1 * val.momentum + (1 - self.beta1) * val.grad
            val.velocity = self.beta2 * val.velocity + (1 - self.beta2) * val.grad ** 2
            m_hat = val.momentum / (1 - self.beta1 ** self.t)
            v_hat = val.velocity / (1 - self.beta2 ** self.t)
            val.data -= self.lr * m_hat / ((v_hat ** 0.5) + self.eps)