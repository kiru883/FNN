from .activate_functions import *
from .loss_functions import *


# activate functions
activates = {'logistic':(logistic, logisticDerivative),
             'relu':(relu, reluDerivative)}

# loss functions
losses = {'multiclassEntropy': lossMulticlassEntropyDerivative,
          'MSE': MSEDerivative}