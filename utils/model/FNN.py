import numpy
import time
from .functions.functions_initialization import activates, losses


class FNN:
    """
            Create model.
             layers: list, List with neurons on each layer, last number should be numbers of classes
             alpha: float, Learning rate
             epochs: int, Number of epochs
             activate_type: str, Name of activate function(realized in activate_functions.py, init. in functions.initialization.py)
             loss_type: str, Name of loss function(realized in loss_functions.py, init. in functions.initialization.py)
             batch_size: int,  Number of train obj. in each batch
             softmax_output: bool, Use softmax output layer or not
             bias: bool, Use biases in activation sums
             verbosity: bool, Write train time and time each epoch
    """

    def __init__(self, layers, alpha=0.1, epochs=10, activate_type='logistic',
                 loss_type='multiclassEntropy', batch_size=None, softmax_output=True,
                 bias=True, verbosity=True):
        self.__with_softmax_output = softmax_output
        self.__verbosity = verbosity
        self.__alpha = alpha
        self.__epochs = epochs

    # set activations(functions and derivative)
        self.__activate_function = activates[activate_type][0]
        self.__activate_function_derivative = activates[activate_type][1]

    # setup loss function
        self.__loss_function_derivative = losses[loss_type]

    # neurons on each layer
        self.__neurons = layers

    # get weights
        self.__variance_scaling(layers)

    # get biases if hyperparam. is true
        self.__with_biases = bias
        self.biases = [numpy.zeros((1, layers[neurons])) for neurons in range(1, len(layers))]

    # get batch size
        self.batch_size = batch_size

    # train model, X and y is horizontal vector(size is (1,))
    def fit(self, X, y):
        """
        Train the model.
        :type X: numpy.array
        :param X: Horizontal vector with train objects, have (-1,) shape
        :type y: numpy.array
        :param y: Horizontal vector with train labels, have (-1,) shape
        """

        if self.__verbosity:
            print("Training started, ", time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))

        if self.batch_size == 'batch':
            self.batch_size = X.shape[0]

        for epoch in range(self.__epochs):
            # need for time of each epoch
            time_start = time.time()

            indexes = numpy.random.permutation(X.shape[0])
            X_shuffled = X[indexes]
            y_shuffled = y[indexes]

            batches = []
            for ind in range(0, X.shape[0], self.batch_size):
                batches.append(list(map(lambda x: (x[0].reshape(1, -1), x[1]),
                                        zip(X_shuffled[ind:ind + self.batch_size],
                               y_shuffled[ind:ind + self.batch_size]))))

            for batch in batches:
                gradient = self.__get_batch_gradient(batch)

                # update weights and biases
                self.weights = [self.weights[layer] - self.__alpha * gradient[0][layer]
                                for layer in range(len(self.__neurons)-1)]
                if self.__with_biases:
                    self.biases = [self.biases[layer] - self.__alpha * gradient[1][layer]
                                   for layer in range(len(self.__neurons)-1)]

            if self.__verbosity:
                print(f"Epoch {epoch+1} is ended, time has passed: {round(time.time() - time_start, 2)} sec.")

    # return predict probabilities for X's
    def predict_proba(self, X):
        """
        Return list of probabilities for each class of train object(s).
        :type X: numpy.array
        :param X: Horizontal vector with train objects, have (-1,) shape
        :return: list of size len(X)
        """

        predicts = []
        for obj in X:
            neurons_signals = self.__feedforward(obj.reshape(1, -1))
            predicts.append(neurons_signals[-1][1])

        return predicts

    # get avg gradient for batch
    def __get_batch_gradient(self, batch):
        # initialize weight and biases matrix
        grad_w = [numpy.zeros_like(array) for array in self.weights]
        grad_b = [numpy.zeros_like(array) for array in self.biases]

        # get gradient for train object
        for X_obj, y_obj in batch:
            neurons_signals = self.__feedforward(X_obj)
            y_vector = numpy.zeros((1, self.__neurons[-1]))
            y_vector[:, y_obj] = 1

            # backpropogation, error on last layer(depending on hyperparam.).
            if self.__with_softmax_output:
                dl_ds = numpy.dot(self.__loss_function_derivative(y_vector, neurons_signals[-1][1]),
                                  self.__softmax_derivative(neurons_signals[-1][1]))
            else:
                dl_ds = numpy.multiply(self.__loss_function_derivative(y_vector, neurons_signals[-1][1]),
                                       self.__activate_function_derivative(neurons_signals[-1][1]))

            # backpropogation
            for layer in range(1, len(self.__neurons)):
                if layer != 1:
                    dl_ds = dl_da * self.__activate_function_derivative(neurons_signals[-layer][0])

                if self.__with_biases:
                    grad_b[-layer] += dl_ds

                grad_w[-layer] += numpy.dot(numpy.transpose(neurons_signals[-layer-1][1]), dl_ds)
                dl_da = numpy.transpose(numpy.dot(self.weights[-layer], numpy.transpose(dl_ds)))

        # get avg. gradient
        grad_w = list(map(lambda x: x/self.batch_size, grad_w))
        if self.__with_biases:
            grad_b = list(map(lambda x: x/self.batch_size, grad_b))

        return grad_w, grad_b

    # get neurons sums and activations for each layer, obj is horizontal feature-vector
    def __feedforward(self, obj):
        activations = obj
        neurons_signals = [(None, activations)]     # sum and activations on first layer

        for layer in range(len(self.__neurons) - 2):
            sums = numpy.dot(activations, self.weights[layer])
            if self.__with_biases:
                sums += self.biases[layer]

            activations = self.__activate_function(sums)
            neurons_signals.append((sums, activations))

        sums = numpy.dot(activations, self.weights[-1])
        if self.__with_biases:
            sums += self.biases[-1]

        # get activations on last layer(depending on hyperparam.)
        if self.__with_softmax_output:
            activations = self.__softmax(sums)
        else:
            activations = self.__activate_function(sums)
        neurons_signals.append((sums, activations))

        return neurons_signals

    # softMax for last layer
    def __softmax(self, arg):
        a = arg - arg.min()
        return numpy.exp(a)/numpy.sum(numpy.exp(a))

    # softmax derivative
    def __softmax_derivative(self, arg):
        # thank you, @mattpetersen!
        return numpy.diag(arg[0]) - numpy.outer(arg, arg)

    # need for weights initialization
    def __variance_scaling(self, layers):
        self.weights = []
        for neurons in range(1, len(layers)):
            fan_avg = (6 / (layers[neurons - 1] + layers[neurons])) ** (0.5)
            self.weights.append(numpy.random.uniform(-fan_avg, fan_avg, (layers[neurons - 1], layers[neurons])))
