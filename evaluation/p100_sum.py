import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import PermutationImportance
import category_encoders as ce

df = pd.read_csv('./../datasets/plates_100.csv')
y = df['goodPlate']
X0 = df.drop(['goodPlate', 'Unnamed: 0'], axis=1)

nominal_columns = ['color']
features = ['diameter',
[
'color_0','color_1','color_2', 'color_3', 'color_4', 'color_5', 'color_6', 'color_7', 'color_8', 'color_9',
'color_10','color_11','color_12', 'color_13', 'color_14', 'color_15', 'color_16', 'color_17', 'color_18', 'color_19',
'color_20','color_21','color_22', 'color_23', 'color_24', 'color_25', 'color_26', 'color_27', 'color_28', 'color_29',
'color_30','color_31','color_32', 'color_33', 'color_34', 'color_35', 'color_36', 'color_37', 'color_38', 'color_39',
'color_40','color_41','color_42', 'color_43', 'color_44', 'color_45', 'color_46', 'color_47', 'color_48', 'color_49',
'color_50','color_51','color_52', 'color_53', 'color_54', 'color_55', 'color_56', 'color_57', 'color_58', 'color_59',
'color_60','color_61','color_62', 'color_63', 'color_64', 'color_65', 'color_66', 'color_67', 'color_68', 'color_69',
'color_70','color_71','color_72', 'color_73', 'color_74', 'color_75', 'color_76', 'color_77', 'color_78', 'color_79',
'color_80','color_81','color_82', 'color_83', 'color_84', 'color_85', 'color_86', 'color_87', 'color_88', 'color_89',
'color_90','color_91','color_92', 'color_93', 'color_94', 'color_95', 'color_96', 'color_97', 'color_98'
]]

encoder = ce.SumEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
# Permutaiton
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(X_train, y_train)
    pi_knn = PermutationImportance.PermutationImportance(clf_knn)
    pi_knn.fit(X_test, y_test, features=features)
    pi_knn.store('p100-sum-knn')
    # KNN
    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train, y_train)
    pi_rfc = PermutationImportance.PermutationImportance(clf_rfc)
    pi_rfc.fit(X_test, y_test, features=features)
    pi_rfc.store('p100-sum-rfc')
    # GNB
    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train, y_train)
    pi_gnb = PermutationImportance.PermutationImportance(clf_gnb)
    pi_gnb.fit(X_test, y_test, features=features)
    pi_gnb.store('p100-sum-gnb')


# ACC SCORE
# KNN
clf_knn_2 = KNeighborsClassifier(n_neighbors=3)
acc_scores_knn = cross_val_score(clf_knn_2, X, y, cv=5)
acc_scores_knn = np.append(acc_scores_knn, np.mean(acc_scores_knn))
np.savetxt('./acc/p100-sum-knn.txt', acc_scores_knn, fmt='%1.3f')
# RFC
clf_rfc_2 = RandomForestClassifier()
acc_scores_rfc = cross_val_score(clf_rfc_2, X, y, cv=5)
acc_scores_rfc = np.append(acc_scores_rfc, np.mean(acc_scores_rfc))
np.savetxt('./acc/p100-sum-rfc.txt', acc_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_2 = GaussianNB()
acc_scores_gnb = cross_val_score(clf_gnb_2, X, y, cv=5)
acc_scores_gnb = np.append(acc_scores_gnb, np.mean(acc_scores_gnb))
np.savetxt('./acc/p100-sum-gnb.txt', acc_scores_gnb, fmt='%1.3f')


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
np.savetxt('./f1/p100-sum-knn.txt', f1_scores_knn, fmt='%1.3f')
# RFC
clf_rfc_3 = RandomForestClassifier()
f1_scores_rfc = cross_val_f1_score(clf_rfc_3, X, y, cv=5)
f1_scores_rfc = np.append(f1_scores_rfc, np.mean(f1_scores_rfc))
np.savetxt('./f1/p100-sum-rfc.txt', f1_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_3 = GaussianNB()
f1_scores_gnb = cross_val_f1_score(clf_gnb_3, X, y, cv=5)
f1_scores_gnb = np.append(f1_scores_gnb, np.mean(f1_scores_gnb))
np.savetxt('./f1/p100-sum-gnb.txt', f1_scores_gnb, fmt='%1.3f')
