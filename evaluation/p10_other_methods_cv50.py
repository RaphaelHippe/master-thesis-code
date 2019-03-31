import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import PermutationImportance

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

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

df = pd.read_csv('./../datasets/autoencoded-p10_1/p10_1_not_scaled_cosine_proximity_tanh_1.csv')
y = df['y']
X = df.drop(['y', 'Unnamed: 0'], axis=1)
features = ['0', 'diameter']
results.append(doAccuracyTests(X, y, 'autoencoder_c_only'))
doPermutationTests(X, y, features, 'autoencoder_c_only')


df = pd.read_csv('./../datasets/autoencoded-p10/p10_10c_150_not_scaled_categorical_crossentropy_sigmoid.csv')
y = df['y']
X = df.drop(['y', 'Unnamed: 0'], axis=1)
features = ['a1', 'a2']
results.append(doAccuracyTests(X, y, 'autoencoder'))


df = pd.read_csv('./../datasets/dummy_pca/p10/p10_n2.csv')
dfy = pd.read_csv('./../datasets/plates_10.csv')
y = dfy['goodPlate']
X = df.drop(['Unnamed: 0'], axis=1)
features = ['pc0', 'pc1']
results.append(doAccuracyTests(X, y, 'pca2'))

df = pd.read_csv('./../datasets/dummy_pca/p10/p10_n11.csv')
dfy = pd.read_csv('./../datasets/plates_10.csv')
y = dfy['goodPlate']
X = df.drop(['Unnamed: 0'], axis=1)
features = ['pc0', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10']
results.append(doAccuracyTests(X, y, 'pca11'))

df = pd.DataFrame(results, columns=['encoding', 'knn', 'rfc', 'gnb'])
df.to_csv('./acc/p10_cv50_other methods.csv')
