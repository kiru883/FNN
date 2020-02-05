from keras.datasets import mnist
from utils.model.FNN import FNN


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
X_train = (x_train / 255)
X_test = (x_test /255)

clf = FNN(layers=[784, 30, 10], epochs=10, batch_size=10, activate_type='logistic',
          loss_type='mse', softmax_output=False, alpha=0.1, bias=True)
clf.fit(X_train, y_train)

predicts = clf.predict_proba(X_test)
res = 0
for ind in range(len(predicts)):
    if predicts[ind].argmax() == y_test[ind]:
        res += 1
print(f"{res} / 10000")


