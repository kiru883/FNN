import numpy
from complex.functions.init import activates, losses


class FNN:

  def __init__(self, layers, alpha=0.1, epochs=10,activate_type='logistic',
               loss_type='multiclassEntropy', gradient_type='batch',
               batch_size=None, bias=True):

    self.__alpha = alpha
    self.__epochs = epochs

    # setup activate function
    self.__activate_function = activates[activate_type][0]
    self.__activate_function_derivative = activates[activate_type][1]

    # setup loss function
    self.__loss_function_derivative = losses[]

    # neurons on each layer
    self.__neurons = layers

    # get weights
    self.weights = [numpy.random.normal(size=(layers[neurons - 1], layers[neurons]))
                    for neurons in range(1, len(layers))]

    # get biases if hyperparam. is true
    self.__with_biases = bias
    self.biases = [numpy.random.normal(size=(1, layers[neurons])) for neurons in range(1, len(layers))]

    # get batch size(depending to gradient type)
    if gradient_type == 'batch':
      self.gradient_type = gradient_type
    elif gradient_type == 'stochastic':
      self.gradient_type = gradient_type
      self.batch_size = 1
    elif gradient_type == 'minibatch':
      self.gradient_type = gradient_type
      self.batch_size = batch_size
    else:
      raise Exception("Bad 'gradient_type'")

  #train model
  def fit(self, X, y):
    if self.gradient_type == 'batch':
      self.batch_size = X.shape[0]

    for epoch in range(self.__epochs):
      X_shuffled = numpy.array(X)
      y_shuffled = numpy.array(y)
      indexes = numpy.random.permutation(X_shuffled.shape[0)]
      X_shuffled = X_shuffled[indexes]
      y_shuffled = y_shuffled[indexes]
      batches = [([X_shuffled[ind:ind + self.batch_size], y_shuffled[ind:ind + self.batch_size])]
                 for ind in range(0, X.shape[0], self.batch_size)]

      for batch in batches:
        gradient = self.__getBatchGradient(batch)
        self.weights -= gradient[0]###
        self.biases -= gradient[1]###

  #get avg gradient for batch
  def __getBatchGradient(self, batch):
    #initialize weight and biases matrix(optional)
    grad_w = [numpy.zeros_like(array) for array in self.weights]
    if self.__with_biases:
      grad_b = [numpy.zeros_like(array) for array in self.biases]

    #get gradient for train object
    for X_obj, y_obj in batch:
      neurons_activity = self.__feedforward(X_obj)
      softmax_prob = self.__softmax(neurons_activity[-1])
      y_vector = numpy.zeros((1, self.__neurons[-1]))
      y_vector[y_obj] = 1

      #backpropogation, Der. loss/Der. activ.
      dl_da = numpy.dot(self.__loss_function_derivative(y_vector, softmax_prob), self.__softmaxDerivative(softmax_prob))
      for

    return grad_w, grad_b

  #get neurons activity
  def __feedforward(self, obj):
    neurons_activity = [obj]
    for layer in range(len(self.__neurons) - 1):
      if self.__with_biases:
        layer_activity = self.__activate_function(numpy.dot(neurons_activity[layer], self.weights[layer])
                                                + self.biases[layer])
      else:
        layer_activity = self.__activate_function(numpy.dot(neurons_activity[layer], self.weights[layer]))
      neurons_activity.append(layer_activity)
    return neurons_activity

  #softMax for last layer
  def __softmax(self, arg):
    a = arg - arg.min()
    return numpy.exp(a)/numpy.sum(numpy.exp(a))
  #softmax derivative
  def __softmaxDerivative(self, arg):
    #thank you, @mattpetersen!
    return numpy.diag(arg) - numpy.outer(arg, arg)