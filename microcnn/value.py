"""
The datatype definition for the neuron gradient operations
"""


class Value:
    def __init__(self, data, children=[], op=None, grad=0.0, momentum= 0.0, velocity=0.0):
        self.data = data
        self.children = children
        self.op = op
        self.grad = grad
        self.momentum = 0.0
        self.velocity = 0.0

    def __repr__(self):
        return (
            f"Value(data: {self.data:.3f}, "
            + f"grad: {self.grad:.3f}, "
            + f"op: {self.op}"
            + (
                f", children: {[f'{c.data:.3f}' for c in self.children]})"
                if self.children
                else ")"
            )
        )

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(data=self.data + other.data, children=[self, other], op="+")
        return result

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(data=self.data * other.data, children=[self, other], op="*")
        return result

    def backward(self):
        if self.op == "+":
            self.children[0].grad += self.grad
            self.children[1].grad += self.grad
        elif self.op == "*":
            self.children[0].grad += self.grad * self.children[1].data
            self.children[1].grad += self.grad * self.children[0].data
