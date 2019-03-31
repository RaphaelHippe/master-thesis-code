import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import PermutationImportance
import category_encoders as ce

nominal_columns = ['color']

features = ['diameter', [
'color_#000019','color_#000033','color_#00004c','color_#000066','color_#00007f',
'color_#000099','color_#0000b2','color_#0000cc','color_#0000e5','color_#0000ff',
'color_#000c00','color_#001900','color_#001919','color_#002600','color_#003300',
'color_#003333','color_#004000','color_#004c00','color_#004c4c','color_#005900',
'color_#006600','color_#006666','color_#007300','color_#007f7f','color_#008000',
'color_#009999','color_#00b2b2','color_#00cccc','color_#00e5e5','color_#00ffff',
'color_#0c000c','color_#100404','color_#170d17','color_#190000','color_#190019',
'color_#191000','color_#191314','color_#191900','color_#210808','color_#260026',
'color_#2f1a2f','color_#310c0c','color_#330000','color_#330033','color_#332100',
'color_#332628','color_#333300','color_#400040','color_#421010','color_#472747',
'color_#4c0000','color_#4c004c','color_#4c3100','color_#4c393c','color_#4c4c00',
'color_#521515','color_#590059','color_#5f345f','color_#631919','color_#660000',
'color_#660066','color_#664200','color_#664c51','color_#666600','color_#730073',
'color_#731d1d','color_#774177','color_#7f0000','color_#7f5200','color_#7f6065',
'color_#7f7f00','color_#800080','color_#842121','color_#8e4e8e','color_#942525',
'color_#990000','color_#996300','color_#997379','color_#999900','color_#a52a2a',
'color_#a65ba6','color_#b20000','color_#b27300','color_#b2868e','color_#b2b200',
'color_#be68be','color_#cc0000','color_#cc8400','color_#cc99a2','color_#cccc00',
'color_#d675d6','color_#e50000','color_#e59400','color_#e5acb6','color_#e5e500',
'color_#ee82ee','color_#ff0000','color_#ffa500','color_#ffc0cb','color_#ffff00',
]]


df = pd.read_csv('./../datasets/plates_100.csv')
y = df['goodPlate']
X = df.drop(['goodPlate', 'Unnamed: 0'], axis=1)

# ENCODING
X = pd.get_dummies(X)
# Permutaiton
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(X_train, y_train)
    pi_knn = PermutationImportance.PermutationImportance(clf_knn)
    pi_knn.fit(X_test, y_test, features=features)
    pi_knn.store('p100-dummy-knn')
    # RFC
    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train, y_train)
    pi_rfc = PermutationImportance.PermutationImportance(clf_rfc)
    pi_rfc.fit(X_test, y_test, features=features)
    pi_rfc.store('p100-dummy-rfc')
    # GNB
    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train, y_train)
    pi_gnb = PermutationImportance.PermutationImportance(clf_gnb)
    pi_gnb.fit(X_test, y_test, features=features)
    pi_gnb.store('p100-dummy-gnb')

# ACC SCORE
# KNN
clf_knn_2 = KNeighborsClassifier(n_neighbors=3)
acc_scores_knn = cross_val_score(clf_knn_2, X, y, cv=5)
acc_scores_knn = np.append(acc_scores_knn, np.mean(acc_scores_knn))
np.savetxt('./acc/p100-dummy-knn.txt', acc_scores_knn, fmt='%1.3f')
# RFC
clf_rfc_2 = RandomForestClassifier()
acc_scores_rfc = cross_val_score(clf_rfc_2, X, y, cv=5)
acc_scores_rfc = np.append(acc_scores_rfc, np.mean(acc_scores_rfc))
np.savetxt('./acc/p100-dummy-rfc.txt', acc_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_2 = GaussianNB()
acc_scores_gnb = cross_val_score(clf_gnb_2, X, y, cv=5)
acc_scores_gnb = np.append(acc_scores_gnb, np.mean(acc_scores_gnb))
np.savetxt('./acc/p100-dummy-gnb.txt', acc_scores_gnb, fmt='%1.3f')


# F1 score
from sklearn.metrics import f1_score

def cross_val_f1_score(clf, X, y, cv=5):
    scores = []
    for _ in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(f1_score(y_test, y_pred, average='micro'))
    return scores

# KNN
clf_knn_3 = KNeighborsClassifier(n_neighbors=3)
f1_scores_knn = cross_val_f1_score(clf_knn_3, X, y, cv=5)
f1_scores_knn = np.append(f1_scores_knn, np.mean(f1_scores_knn))
np.savetxt('./f1/p100-dummy-knn.txt', f1_scores_knn, fmt='%1.3f')
# RFC
clf_rfc_3 = RandomForestClassifier()
f1_scores_rfc = cross_val_f1_score(clf_rfc_3, X, y, cv=5)
f1_scores_rfc = np.append(f1_scores_rfc, np.mean(f1_scores_rfc))
np.savetxt('./f1/p100-dummy-rfc.txt', f1_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_3 = GaussianNB()
f1_scores_gnb = cross_val_f1_score(clf_gnb_3, X, y, cv=5)
f1_scores_gnb = np.append(f1_scores_gnb, np.mean(f1_scores_gnb))
np.savetxt('./f1/p100-dummy-gnb.txt', f1_scores_gnb, fmt='%1.3f')
