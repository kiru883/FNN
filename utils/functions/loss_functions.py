import numpy
# all functions must return horizontal vector(1, -1)


# multiclass entropy
def multiclassEntropyDerivative(y, y_):
    return -y / y_

# mse
def mseDerivative(y, y_):
    return y_ - y

# helinger
def helingerDerivative(y, y_):
    return (numpy.sqrt(y_) - numpy.sqrt(y)) / numpy.sqrt(2*y_)

# kullback
def kullbackDerivative(y, y_):
    return (y_ - y) / y_

