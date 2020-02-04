import numpy
# all functions must return horizontal vector


# logistic function
def logistic(arg):
    return 1.0 / (1.0 + numpy.exp(-arg))

def logisticDerivative(arg):
    return numpy.exp(-arg) / ((numpy.exp(-arg) + 1.0) ** 2.0)


# relu
def relu(arg):
    return numpy.where(arg <= 0, 0, arg)

def reluDerivative(arg):
    return numpy.where(arg <= 0, 0, 1)


