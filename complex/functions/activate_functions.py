import numpy
# all functions must return horizontal vector


# logistic function
def activationLogistic(arg):
    return 1 / (1 + numpy.exp(-arg))


def activationLogisticDerivative(arg):
    return numpy.exp(arg) / ((numpy.exp(-arg) + 1) ** 2)

