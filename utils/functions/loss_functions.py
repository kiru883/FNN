import numpy
# all functions must return horizontal vector


# multiclass entropy
def lossMulticlassEntropy(y, y_):
    return -y * numpy.log(y_)

def lossMulticlassEntropyDerivative(y, y_):
    return -y / y_


# MSE
def MSEDerivative(y, y_):
    return y_ - y