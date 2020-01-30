from .activate_functions import *
from .loss_functions import *

#activate functions
activates = {'logistic':(activationLogistic, activationLogisticDerivative)}

#loss finctions
losses = {'multiclassEntropy':(lossMulticlassEntropy, lossMulticlassEntropyDerivative)}