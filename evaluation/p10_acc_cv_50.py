import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import PermutationImportance
import category_encoders as ce

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


df = pd.read_csv('./../datasets/plates_10.csv')
y = df['goodPlate']
X0 = df.drop(['goodPlate', 'Unnamed: 0'], axis=1)

nominal_columns = ['color']



def doPermutationTests(X, y, features, encoding):
    columns = ['x1_knn','x2_knn','x1_rfc','x2_rfc','x1_gnb','x2_gnb']
    data = []
    for _ in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        # KNN
        clf_knn = KNeighborsClassifier(n_neighbors=3)
        clf_knn.fit(X_train, y_train)
        pi_knn = PermutationImportance.PermutationImportance(clf_knn)
        pi_knn.fit(X_test, y_test, features=features)
        pi_knn_df = pi_knn.get_weights()
        # KNN
        clf_rfc = RandomForestClassifier()
        clf_rfc.fit(X_train, y_train)
        pi_rfc = PermutationImportance.PermutationImportance(clf_rfc)
        pi_rfc.fit(X_test, y_test, features=features)
        pi_rfc_df = pi_rfc.get_weights()
        # GNB
        clf_gnb = GaussianNB()
        clf_gnb.fit(X_train, y_train)
        pi_gnb = PermutationImportance.PermutationImportance(clf_gnb)
        pi_gnb.fit(X_test, y_test, features=features)
        pi_gnb_df = pi_gnb.get_weights()

        pi_knn_df.loc[pi_knn_df['feature'] == 'diameter', 'feature'] = 'x1'
        pi_knn_df.loc[pi_knn_df['feature'] != 'x1', 'feature'] = 'x2'

        pi_rfc_df.loc[pi_rfc_df['feature'] == 'diameter', 'feature'] = 'x1'
        pi_rfc_df.loc[pi_rfc_df['feature'] != 'x1', 'feature'] = 'x2'

        pi_gnb_df.loc[pi_gnb_df['feature'] == 'diameter', 'feature'] = 'x1'
        pi_gnb_df.loc[pi_gnb_df['feature'] != 'x1', 'feature'] = 'x2'

        data.append([
            pi_knn_df['acc_weight'].loc[pi_knn_df['feature'] == 'x1'].iloc[0],
            pi_knn_df['acc_weight'].loc[pi_knn_df['feature'] == 'x2'].iloc[0],
            pi_rfc_df['acc_weight'].loc[pi_rfc_df['feature'] == 'x1'].iloc[0],
            pi_rfc_df['acc_weight'].loc[pi_rfc_df['feature'] == 'x2'].iloc[0],
            pi_gnb_df['acc_weight'].loc[pi_gnb_df['feature'] == 'x1'].iloc[0],
            pi_gnb_df['acc_weight'].loc[pi_gnb_df['feature'] == 'x2'].iloc[0]
        ])

    df = pd.DataFrame(data, columns=columns)
    df.boxplot(rot=90, figsize=cm2inch(8, 8))
    plt.subplots_adjust(bottom=0.2)
    plt.ylim(-1, 1)
    plt.savefig("./images/p10-permutation-importance-boxplot-{}.svg".format(encoding))
    plt.clf()



def doAccuracyTests(X,y, encoding):
    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    acc_scores_knn = cross_val_score(clf_knn, X, y, cv=50)
    # RFC
    clf_rfc = RandomForestClassifier()
    acc_scores_rfc = cross_val_score(clf_rfc, X, y, cv=50)
    # GNB
    clf_gnb = GaussianNB()
    acc_scores_gnb = cross_val_score(clf_gnb, X, y, cv=50)
    return [encoding, np.mean(acc_scores_knn), np.mean(acc_scores_rfc), np.mean(acc_scores_gnb)]

results = []
encoder = ce.OneHotEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'onehot'))
features = ['diameter', ['color_1',
'color_2', 'color_3', 'color_4', 'color_5', 'color_6',
'color_7', 'color_8', 'color_9', 'color_10', 'color_-1']]
doPermutationTests(X, y, features, 'onehot')

X = pd.get_dummies(X0)
results.append(doAccuracyTests(X, y, 'dummy'))
features = ['diameter', ['color_blue','color_brown',
'color_cyan', 'color_green', 'color_orange', 'color_pink',
'color_purple', 'color_red','color_violet', 'color_yellow']]
doPermutationTests(X, y, features, 'dummy')

encoder = ce.BackwardDifferenceEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'difference'))
features = ['diameter', ['color_0','color_1',
'color_2', 'color_3', 'color_4', 'color_5', 'color_6', 'color_7', 'color_8']]
doPermutationTests(X, y, features, 'difference')

encoder = ce.BaseNEncoder(base=3, cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'basen'))
features = ['diameter', ['color_0','color_1',
'color_2', 'color_3']]
doPermutationTests(X, y, features, 'basen')

encoder = ce.BinaryEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'binary'))
features = ['diameter', ['color_0','color_1',
'color_2', 'color_3', 'color_4']]
doPermutationTests(X, y, features, 'binary')

encoder = ce.HelmertEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'helmert'))
features = ['diameter', ['color_0', 'color_1',
'color_2', 'color_3', 'color_4', 'color_5', 'color_6',
'color_7', 'color_8']]
doPermutationTests(X, y, features, 'helmert')

encoder = ce.SumEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'sum'))
features = ['diameter', ['color_0', 'color_1',
'color_2', 'color_3', 'color_4', 'color_5', 'color_6',
'color_7', 'color_8']]
doPermutationTests(X, y, features, 'sum')

encoder = ce.LeaveOneOutEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'leaveoneout'))
features = ['diameter', 'color']
doPermutationTests(X, y, features, 'leaveoneout')

encoder = ce.TargetEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'target'))
features = ['diameter', 'color']
doPermutationTests(X, y, features, 'target')

encoder = ce.OrdinalEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'ordinal'))
features = ['diameter', 'color']
doPermutationTests(X, y, features, 'ordinal')

encoder = ce.WOEEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'woe'))
features = ['diameter', 'color']
doPermutationTests(X, y, features, 'woe')


df = pd.DataFrame(results, columns=['encoding', 'knn', 'rfc', 'gnb'])
df.to_csv('./acc/p10_cv50.csv')
