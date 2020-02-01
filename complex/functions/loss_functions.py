import numpy
# all functions must return horizontal vector


# multiclass entropy
def lossMulticlassEntropy(y, y_):
    return -y * numpy.log(y_)


def lossMulticlassEntropyDerivative(y, y_):
    return -y / y_