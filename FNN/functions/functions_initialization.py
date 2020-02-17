from .activate_functions import *
from .loss_functions import *


# activate functions
activates = {'logistic': (logistic, logisticDerivative),
             'relu': (relu, reluDerivative),
             'tanh': (tanh, tanhDerivative),
             'softsign': (softsign, softsignDerivative),
             'elu': (elu, eluDerivative),
             'softplus': (softplus, softplusDerivative),
             'lrelu': (lrelu, lreluDerivative),
             'swish': (swish, swishDerivative)
             }

# loss functions
losses = {'multiclassEntropy': multiclassEntropyDerivative,
          'mse': mseDerivative,
          'helinger': helingerDerivative,
          'kullback': kullbackDerivative
          }
