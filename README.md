# Feedforward neural network for **binary / multiclass classification**

Created for **educational purposes** network with dense layers for classification tasks.

# How to install
```
>> git clone https://github.com/kiru883/FNN.git
>> cd /FNN
>> pip install -r requirements.txt 
```

# Usage example
An example of use is described in **example.py.**

```
model = FNN(
    layers=[784, 30, 10],          <-- Number of neurons, 784 number of inputs, 30 - num. of one hidden 
                                       layer neurons(for example FNN have three hiden layers with random number
                                       of neurons, then parameter 'layers' be equal [784, 30, 50, 70, 30]),
                                       10 - number of output neurons. 
    epochs=10,                     <-- Number of epochs.
    batch_size=10,                 <-- Number of samples in batch.
    activate_type='logistic',      <-- Activate function for neurons on hidden 
                                       layer 
    loss_type='mse',               <-- Loss function.
    softmax_output=False,          <-- Use softmax output or not.
    alpha=0.1,                     <-- Learning rate.
    bias=True                      <-- Use bias in activation functions or not.
)
...
model.fit(X, y)                    <-- X must be numpy array with size (N_samples, n_features), y must be array of 
                                       shape (, N_samples) with class labels(for example [1, ..., 9, 0]).
model.predict_proba(X)
```

# Activation functions and losses
#### Activations:
1. Logistic
2. ReLU
3. Tanh
4. Softsign
5. ELU
6. Softplus
7. LReLU
8. Swish
#### Losses:
1. MSE
2. MulticlassEntropy
3. Helinger
4. kullback