"""
It contains the activation classes for theano variables
"""
import theano
import theano.tensor as tensor
import numpy as np
class Activation(object):
    def __init__(self, act_params):
        pass
class Relu(Activation):
    def __init__(self, act_params=None):
        self.name = 'relu'
    @classmethod
    def __call__(self, X):
        return theano.tensor.switch(X<0,0,X)
    def __str__(self):
        return 'Rectified Linear Unit (y = x if x >= 0)'

class Relu2(Activation):
    def __init__(self, act_params=None):
        self.name = 'relu2'
    @classmethod
    def __call__(self, X):
        return theano.tensor.switch(X>0,X,0)
    def __str__(self):
        return 'Rectified Linear Unit2: when y = x if x > 0'
        
class Abs(Activation):
    def __init__(self, act_params=None):
        self.name = 'abs'
    @classmethod
    def __call__(self, X):
        return abs(X)
    def __str__(self):
        return 'Absolute Unit'
class BinaryCrossEntropy(Activation):
    def __init__(self, act_params=None):
        self.name = 'binary_crossentropy'
    @classmethod
    def __call__(self, X_pred,X_gt):
        return tensor.nnet.binary_crossentropy(X_pred, X_gt)
    def __str__(self):
        return 'Binary CrossEntropy'
class Tanh(Activation):
    """
    tanh[a,b] = a * tanh(b * x)
    """
    def __init__(self, act_params=None):
        self.name='tanh'
        self.a = theano.shared(np.cast[theano.config.floatX](act_params[0]))
        self.b = theano.shared(np.cast[theano.config.floatX](act_params[1]))
    def __call__(self, X):
        return self.a * tensor.tanh(self.b * X)
class Sigmoid(Activation):
    """
    logistic(x) = 1/ (1 + e^(-x))
    """
    def __init__(self, act_params=None):
        self.name = 'sigmoid'
    @classmethod
    def __call__(self, X):
        return tensor.nnet.sigmoid(X)
class Linear(Activation):
    """
    linear[a,b](x) =  a * x + b
    """
    def __init__(self, act_params=None):
        self.name = 'linear'
        self.do_calc = False
        if len(act_params) == 2:
            self.a = theano.shared(np.cast[theano.config.floatX](act_params[0]))
            self.b = theano.shared(np.cast[theano.config.floatX](act_params[1]))
            if float(act_params[0]) != 1 or float(act_params[1])!=0:
                self.do_calc = True
        else:
            self.a = self.b = None
    def __call__(self, X):
        if self.do_calc:
            return self.a * X + self.b
        else:
            return X

class Softmax(Activation):
    """
    softmax
    """
    def __init__(self, act_params= None):
        self.name = 'softmax'
    def __call__(self, X):
        e_X = tensor.exp(X - X.max(axis=1, keepdims=True))
        out = e_X / e_X.sum(axis=1, keepdims=True)
        return out
def make_actfunc(name, params):
    return act_dic[name](params)



act_dic = {'relu':Relu, 'relu2':Relu2, 'abs':Abs, 'binary_crossentropy':BinaryCrossEntropy,
           'tanh':Tanh, 'sigmoid':Sigmoid, 'linear':Linear, 'softmax':Softmax}