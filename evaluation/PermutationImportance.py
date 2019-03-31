import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os

class PermutationImportance(object):
    """docstring for ."""
    def __init__(self, clf, n=10):
        self.clf = clf
        self.n = n
        self.base_acc = 0
        self.feature_importance = None

    def fit(self, X_test, y_test, features):
        # get base acc
        self.base_acc = self.clf.score(X_test, y_test)
        # permutate + calc difference
        feature_importance = []
        for feature in features:
            X_temp = X_test.copy()
            acc_temp = 0
            if isinstance(feature,(list,)):
                for _ in range(self.n):
                    # permutation
                    df_temp = X_temp[feature].copy()
                    df_shuffled = shuffle(df_temp)
                    for col in df_shuffled.columns:
                        X_temp[col] = df_shuffled[col]
                    # accuracy
                    acc_temp = acc_temp + self.clf.score(X_temp, y_test)
                acc_temp = acc_temp / self.n
                feature_importance.append([feature, self.base_acc - acc_temp])
            else:
                for _ in range(self.n):
                    # permutation
                    X_temp[feature] = np.random.permutation(X_temp[feature].values)
                    # accuracy
                    acc_temp = acc_temp + self.clf.score(X_temp, y_test)
                acc_temp = acc_temp / self.n
                feature_importance.append([feature, self.base_acc - acc_temp])
        self.feature_importance = pd.DataFrame(feature_importance, columns=['feature', 'acc_weight'])


    def show_weights(self, n=5):
        print(self.feature_importance.head(n=n))

    def store(self, file):
        filename = './results/{}.csv'.format(file)
        if os.path.isfile(filename):
            with open(filename, 'a') as f:
             (self.feature_importance).to_csv(f, header=False)
        else:
            self.feature_importance.to_csv(filename)

    def get_weights(self):
        return self.feature_importance
