import numpy
from FNNS.utils.functions.functions_initialization import activates, losses


class FNN:

    def __init__(self, layers, alpha=0.1, epochs=10, activate_type='logistic',
                 loss_type='multiclassEntropy', gradient_type='batch',
                 batch_size=None, bias=True):

        self.__alpha = alpha
        self.__epochs = epochs

        self.__activate_function = activates[activate_type][0]
        self.__activate_function_derivative = activates[activate_type][1]

    # setup loss function
        self.__loss_function_derivative = losses[loss_type][1]

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

    # train model, X and y is horizontal vector(size is (1,))
    def fit(self, X, y):
        if self.gradient_type == 'batch':
            self.batch_size = X.shape[0]

        for epoch in range(self.__epochs):
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
                                for layer in range(self.__neurons)]
                if self.__with_biases:
                    self.biases = [self.biases[layer] - self.__alpha * gradient[1][layer]
                                   for layer in range(self.__neurons)]

    # get avg gradient for batch
    def __get_batch_gradient(self, batch):
        # initialize weight and biases matrix(optional)
        grad_w = [numpy.zeros_like(array) for array in self.weights]
        if self.__with_biases:
            grad_b = [numpy.zeros_like(array) for array in self.biases]

        # get gradient for train object
        for X_obj, y_obj in batch:
            neurons_signals = self.__feedforward(X_obj)
            softmax_prob = self.__softmax(neurons_signals[-1][1])
            y_vector = numpy.zeros((1, self.__neurons[-1]))
            y_vector[:, y_obj] = 1

            # backpropogation, Der. loss/Der. activ.
            dl_da = numpy.dot(self.__loss_function_derivative(y_vector, softmax_prob),
                              self.__softmax_derivative(softmax_prob))
            for layer in range(1, len(self.__neurons)):
                print("layer: ", layer)
                # dl_da*da_ds(element-wise multiple) = dl_ds, also bias gradient
                dl_ds = numpy.multiply(dl_da, self.__activate_function_derivative(neurons_signals[-layer][0]))
                print("dlds: ", dl_ds)
                if self.__with_biases:
                    grad_b[-layer] += dl_ds

                print("weights: ", self.weights[-layer])
                print("activ: ", neurons_signals[-layer][1])




                grad_w[-layer] += numpy.dot(numpy.transpose(neurons_signals[-layer-1][1]), dl_ds)
                dl_da = numpy.dot(self.weights[-layer], numpy.transpose(dl_ds))

        # get avg. gradient
        grad_w = list(map(lambda x: x/self.batch_size, grad_w))
        if self.__with_biases:
            grad_b = list(map(lambda x: x/self.batch_size, grad_b))

        return grad_w, grad_b

    # get neurons sums and activations for each layer, obj is horizontal feature-vector
    def __feedforward(self, obj):
        activations = obj
        neurons_signals = []

        for layer in range(len(self.__neurons) - 1):
            if self.__with_biases:
                sums = numpy.dot(activations, self.weights[layer]) + self.biases[layer]
            else:
                sums = numpy.dot(activations, self.weights[layer])
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
        return numpy.diag(arg) - numpy.outer(arg, arg)
