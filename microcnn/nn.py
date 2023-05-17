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
        self.X = None
        self.z = Value(0.0)

    def forward(self, X):
        self.X = X
        output = sum([w.data * x.data for w, x in zip(self.w, X)], self.b.data)
        self.z = Value(output, children=self.w+X+[self.b], op='neuron')
        return self.z
    
    def backward(self):
        for w, x in zip(self.w, self.X):
            w.grad += self.z.grad * x.data
            if x.children:
                x.grad += self.z.grad * w.data
        self.b.grad += self.z.grad

    def parameters(self):
        if self.X:
            return self.w + self.X + [self.b, self.z]
        return self.w + [self.b, self.z]

class Layer:
    def __init__(self, n_inputs, n_neurons, act_fn=None):
        self.n_inputs = n_inputs
        self.act_fn = act_fn
        self.neurons = [Neuron(n_inputs) for i in range(n_neurons)]
        self.act_out = None

    def forward(self, X):
        neuron_out = [neuron.forward(X) for neuron in self.neurons]
        if self.act_fn:
            self.act_out = self.act_fn.forward(neuron_out)
            return self.act_out
        return neuron_out

    def backward(self):
        if self.act_fn:
            self.act_fn.backward()
        for neuron in self.neurons:
            neuron.backward()

    def parameters(self):
        if self.act_fn:
            if self.act_out:
                return [param for neuron in self.neurons for param in neuron.parameters()]+self.act_out
            return [param for neuron in self.neurons for param in neuron.parameters()]
        return [param for neuron in self.neurons for param in neuron.parameters()]


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            assert len(X) == layer.n_inputs, "Size mismatch"
            X = layer.forward(X)
        return X
    
    def backward(self):
        # self.zero_grad()
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0


########################## Loss Functions ##############################


class RMSLoss:
    def forward(self, Y_pred, Y_gt):
        assert len(Y_pred) == len(Y_gt), "Size mismatch"
        self.Y_pred = Y_pred  
        self.Y_diff = [(y_pred.data - y_gt.data) for y_pred, y_gt in zip(Y_pred, Y_gt)]
        self.loss = (sum(y_diff**2 for y_diff in self.Y_diff) / len(Y_pred)) ** 0.5
        self.loss_val = Value(self.loss, children=Y_pred+Y_gt, op='rms_loss', grad=1.0)
        
        return self.loss_val
    
    def backward(self):
        for y_pred, y_diff in zip(self.Y_pred, self.Y_diff):
            y_pred.grad += 2 * y_diff / (self.loss * len(self.Y_pred))
    
    
class CrossEntropyLoss:
    def forward(self, Y_pred, Y_gt):
        assert len(Y_pred) == len(Y_gt), "Size mismatch"
        self.Y_pred = Y_pred
        self.Y_gt = Y_gt
        loss = -sum(y_gt.data * math.log(y_pred.data) 
                    for y_pred, y_gt in zip(Y_pred, Y_gt))
        loss_val = Value(loss, children=Y_pred+Y_gt, op='cce_loss', grad=1.0)
        return loss_val
    
    def backward(self):
        for y_pred, y_gt in zip(self.Y_pred, self.Y_gt):
            y_pred.grad += - y_gt.data / y_pred.data


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
        loss = -sum(y_gt.data * math.log(y_softmax) 
                    for y_gt, y_softmax in zip(Y_gt, self.softmax_data))
        self.loss_val = Value(loss, children=Y_pred+Y_gt, op='softmax_cce_loss')
        return self.loss_val

    def backward(self):
        for y_pred, y_softmax, y_gt in zip(self.Y_pred, self.softmax_data, self.Y_gt):
            # Gradient is softmax outputs - ground truth
            y_pred.grad += y_softmax - y_gt.data
            
            
######################### Activation Functions #########################


class ReLU:
    def forward(self, neuron_outs):
        self.outs = [Value(max(val.data, 0), children=[val], op='relu') 
                for val in neuron_outs]
        return self.outs

    def backward(self):
        for out in self.outs:
            out.children[0].grad += out.grad * (out.data >0)


class LeakyReLU:
    def forward(self, neuron_outs):
        self.outs = [Value(max(val.data, 0.01 * val.data), children=[val], op='l_relu') 
                for val in neuron_outs]
        return self.outs

    def backward(self):
        for out in self.outs:
            out.children[0].grad += out.grad * (1 if out.data > 0 else 0.01)
            

class Tanh:
    def forward(self, neuron_outs):
        self.outs = [Value(math.tanh(val.data), children=[val], op='tanh') 
                     for val in neuron_outs]
        return self.outs
    
    def backward(self):
        for out in self.outs:
            out.children[0].grad += out.grad * (1 - out.data**2)


class Softmax:
    def forward(self, neuron_outs):
        exp_vals = [math.exp(val.data) for val in neuron_outs]
        exp_sum = sum(exp_vals)
        softmax_data = [exp_val / exp_sum for exp_val in exp_vals]

        self.outs = [Value(data, children=[neuron_outs[i]], op='softmax') 
                  for i, data in enumerate(softmax_data)]
        return self.outs

    def backward(self):
        for i, out in enumerate(self.outs):
            for j, _ in enumerate(self.outs):
                if i == j:
                    out.children[0].grad += out.grad * (1 - out.data)
                else:
                    out.children[0].grad -= out.grad * out.data
        

######################### Optimizers ###################################


class SGD:
    def step(self, vals, lr=1e-3, m=0.9):
        for val in vals:
            val.momentum = m * val.momentum + lr * val.grad
            val.data -= val.momentum
