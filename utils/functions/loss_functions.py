import numpy
# all functions must return horizontal vector(1, -1)


# multiclass entropy
def lossMulticlassEntropyDerivative(y, y_):
    return -y / y_

# MSE
def MSEDerivative(y, y_):
    return y_ - y