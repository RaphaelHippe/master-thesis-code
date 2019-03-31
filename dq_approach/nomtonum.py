import ca
import numpy as np
import pandas as pd

class NOMTONUM(object):
    """
    === Params:
    nominal_features: List of all nominal features to be transformed into numerical values.
    === Returns: None
    """
    def __init__(self, nominal_features, ignored_features):

        self.nominal_features = nominal_features
        self.ignored_features = ignored_features
        self.dict = {}
        for nom_feat in self.nominal_features:
            self.dict[nom_feat] = {}

    def checkEigenvals(self, eigenvals, n):
        check = True
        for i, val in enumerate(eigenvals):
            if i < n:
                if val != 1.0:
                    check = False
        return check

    def calcNominalVal(self, coordinates, check, n):
        if check:
            val = 0.
            for j in xrange(n):
                val += coordinates[j]
            return val
        else:
            return coordinates[0]




    # def countValues(self, X):
    #     pass
    #
    # def createCountTable(self, X, feature, other_feature):



    def createCountTablesForFeature(self, X, feature, other_features):
        countsTables = []
        feature_value_names = []
        feature_uniques = X[feature].unique()
        for other_feature in other_features:
            # print(feature, other_feature)
            if not any(other_feature in s for s in self.ignored_features):
                X_counts = X.groupby([feature,other_feature]).size().reset_index().rename(columns={0:'count'})
                other_feature_uniques = X[other_feature].unique()
                table = np.zeros(shape=(len(feature_uniques),len(other_feature_uniques)), dtype=int)
                row_index = 0
                col_index = 0
                for feature_val_unique in feature_uniques:
                    for other_feature_val_unique in other_feature_uniques:
                        vals = X_counts.loc[(X_counts[feature] == feature_val_unique) & (X_counts[other_feature] == other_feature_val_unique), "count"].values
                        if len(vals) > 0:
                            table[row_index][col_index] = vals[0]
                        else:
                            table[row_index][col_index] = 0

                        if col_index == len(table[row_index]) - 1:
                            col_index = 0
                            row_index += 1
                        else:
                            col_index += 1
                countsTables.append(pd.DataFrame(table))
        for name in X[feature].unique():
            feature_value_names.append(name)
        result = pd.concat(countsTables, axis=1)
        return (result, feature_value_names)

    """
    === Params:
    X: The data in form of a pandas DataFrame
    === Returns:
    X_transformed: The data with nominal values transformed to numerical values
    in form of a pandas DataFrame
    """
    def fit_transform(self, X):
        X_transformed = X

        for nom_feat in self.nominal_features:
            table, feature_value_names = self.createCountTablesForFeature(X, nom_feat, np.delete(X.columns, np.where(X.columns == nom_feat)))
            # print(table)
            # print(table.shape)
            # print(np.isnan(table).any())
            # print(np.isinf(table).any())

            myca = ca.CA(table)
            if myca.eigenvals[0] == 0. and myca.eigenvals[1] == 0.:
                raise ValueError('First 2 eigenvalues are 0, therefore encoding fails')

            # n = 2 hardcoded? Correct?
            n = 2
            check = self.checkEigenvals(myca.eigenvals, n)
            for label, dim1, dim2 in zip(feature_value_names, myca.F[:, 0], myca.F[:, 1]):
                X_transformed.loc[X[nom_feat] == label, nom_feat] = self.calcNominalVal([dim1, dim2], check, n)
        return X_transformed
