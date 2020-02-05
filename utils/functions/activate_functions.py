import numpy
# all functions must return horizontal vector(1, -1)


# logistic
def logistic(arg):
    return 1.0 / (1.0 + numpy.exp(-arg))

def logisticDerivative(arg):
    return numpy.exp(-arg) / ((numpy.exp(-arg) + 1.0) ** 2.0)


# relu
def relu(arg):
    return numpy.where(arg <= 0, 0, arg)

def reluDerivative(arg):
    return numpy.where(arg <= 0, 0, 1)


# tanh
def tanh(arg):
    return numpy.tanh(arg)

def tanhDerivative(arg):
    return 1 - ((numpy.tanh(arg))**2)


# bent
def bent(arg):
    return ((numpy.sqrt((arg**2) + 1) - 1) / 2) + arg

def bentDerivative(arg):
    return (arg / (2 * numpy.sqrt((arg**2) + 1))) + 1


# softsign
def softsign(arg):
    return arg / (1 + numpy.abs(arg))

def softsignDerivative(arg):
    return 1 / (1 + numpy.abs(arg))**2


# elu
def elu(arg):
    return numpy.where(arg <= 0, (numpy.exp(arg)-1), arg)

def eluDerivative(arg):
    return numpy.where(arg <= 0, numpy.exp(arg), arg)


# softplus
def softplus(arg):
    return numpy.log(1 + numpy.exp(arg))

def softplusDerivative(arg):
    return 1 / (1 + numpy.exp(-arg))


# lrelu
def lrelu(arg):
    return numpy.where(arg < 0, 0.01*arg, arg)

def lreluDerivative(arg):
    return numpy.where(arg < 0, 0.01, 1)


# swish
def swish(arg):
    return arg / (1-numpy.exp(-arg))

def swishDerivative(arg):
    return arg / (1 - numpy.exp(-arg))




