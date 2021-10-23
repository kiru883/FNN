import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from FNN.FNN import FNN


if __name__ == '__main__':
    # load datasets
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X, y = X.to_numpy(), y.to_numpy().astype(np.int32).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y, shuffle=True)

    X_train = (X_train / 255)
    X_test = (X_test / 255)

    # init. FNN with 30 hiden neurons and 10 output neurons for multiclass classif. (10 classes)
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
