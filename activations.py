from np import *
from functions import *


class Id:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        return x

    def backward(self, delta):
        return delta

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        y = x.copy()
        self.mask = (y <= 0)
        y[self.mask] = 0
        return y

    def backward(self, delta):
        delta[self.mask] = 0
        dx = delta
        return dx

class Tanh:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = np.tanh(x)
        self.y = y
        return y

    def backward(self, delta):
        dx = delta * (1 - self.y ** 2)
        return dx

class Softmax:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = softmax(x)
        self.y = y
        return y

    def backward(self, t):
        batch_size = t.shape[0]
        delta = (self.y - t) / batch_size
        return delta


act = {'Id':Id, 'Relu':Relu, 'Softmax':Softmax, 'Tanh':Tanh}
