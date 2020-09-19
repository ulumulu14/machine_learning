#Partial Dependence Plot


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing


def pdp(classifier, X, features):
    for i, feature in enumerate(features):
        avg_targets = []
        original_column = X[[feature]]
        unique_values = pd.unique(X[feature])

        for value in unique_values:
            column = pd.DataFrame({feature : np.full_like(X[feature], value)})
            X[feature] = column

            targets = classifier.predict(X)
            avg_targets.append(np.average(targets))

        plt.subplot(len(features), 1, i+1)
        plt.plot(unique_values, avg_targets, color="red")
        plt.xlabel(feature)
        plt.ylabel("Partial Dependance")
        plt.subplots_adjust(hspace=0.5)

        X[feature] = original_column

    plt.show()


if __name__ == "__main__":
    X_train = pd.DataFrame({"f1" : [1, 2, 2, 4, 10], "f2" : [3, 4, 5, 1, 15], "f3" : [2, 3, 1, 3, 2]})
    y_train = [3, 4, 3, 3, 10]
    features = ["f1", "f2", "f3"]

    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = LinearRegression()
    model.fit(X_train, y_train)

    pdp(model, X_train, features)
    plt.show()