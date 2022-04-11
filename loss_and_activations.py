import numpy as np
from sympy import Si

class Sigmoid: 
    def __init__(self) -> None:
        self.in_vector = None

    def forward(self, vector):
        self.in_vector = vector
        return 1/(1+np.e**(-vector))

    def backward(self):
        vector = self.in_vector
        # self.in_vector =None
        # return self.forward(vector)*(1-self.forward(vector))
        return 1 # cancelled out by bce derivitive

class Relu: 

    def __init__(self) -> None:
        self.in_vector = None

    def forward(self, vector):
        self.in_vector = vector
        return vector * (vector > 0)

    def backward(self):
        vector = self.in_vector
        # self.in_vector =None
        return vector > 0


class BCE: 

    def __init__(self):
        self.derivitive = None

    def forward(self, predictions, targets):
        # loss is mse and is just a gaugue of model performance. Actual bce is in the derivative
        loss = .5*(predictions - targets)**2
        self.derivative = targets - predictions
        return np.sum(loss)

    def backward(self):
        der = self.derivative
        self.derivative = None
        return der