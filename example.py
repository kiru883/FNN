from keras.datasets import mnist
from utils.FNN import FNN

# load datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
X_train = (x_train / 255)
X_test = (x_test / 255)

# init. model with 30 hiden neurons and 10 output neurons for multiclass classif. (10 classes)
clf = FNN(layers=[784, 30, 10], epochs=10, batch_size=10, activate_type='logistic',
          loss_type='mse', softmax_output=False, alpha=0.1, bias=True)
clf.fit(X_train, y_train)

# get predicts and accuracy
predicts = clf.predict_proba(X_test)
score = 0
for ind in range(len(predicts)):
    if predicts[ind].argmax() == y_test[ind]:
        score += 1
print(f"Accuracy: {score} / {X_test.shape[0]}")
