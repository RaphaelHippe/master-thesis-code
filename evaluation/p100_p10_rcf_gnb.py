import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import PermutationImportance

features = ['diameter', 'color']


df = pd.read_csv('./../datasets/plates_100.csv')
y = df['goodPlate']
X = df.drop(['goodPlate', 'Unnamed: 0'], axis=1)

# Permutaiton
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # RFC
    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train, y_train)
    pi_rfc = PermutationImportance.PermutationImportance(clf_rfc)
    pi_rfc.fit(X_test, y_test, features=features)
    pi_rfc.store('p100-p10-no-encoding-rfc')
    # GNB
    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train, y_train)
    pi_gnb = PermutationImportance.PermutationImportance(clf_gnb)
    pi_gnb.fit(X_test, y_test, features=features)
    pi_gnb.store('p100-p10-no-encoding-gnb')

# ACC SCORE
# RFC
clf_rfc_2 = RandomForestClassifier()
acc_scores_rfc = cross_val_score(clf_rfc_2, X, y, cv=5)
acc_scores_rfc = np.append(acc_scores_rfc, np.mean(acc_scores_rfc))
np.savetxt('./acc/p100-p10-no-encoding-rfc.txt', acc_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_2 = GaussianNB()
acc_scores_gnb = cross_val_score(clf_gnb_2, X, y, cv=5)
acc_scores_gnb = np.append(acc_scores_gnb, np.mean(acc_scores_gnb))
np.savetxt('./acc/p100-p10-no-encoding-gnb.txt', acc_scores_gnb, fmt='%1.3f')


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

# RFC
clf_rfc_3 = RandomForestClassifier()
f1_scores_rfc = cross_val_f1_score(clf_rfc_3, X, y, cv=5)
f1_scores_rfc = np.append(f1_scores_rfc, np.mean(f1_scores_rfc))
np.savetxt('./f1/p100-p10-no-encoding-rfc.txt', f1_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_3 = GaussianNB()
f1_scores_gnb = cross_val_f1_score(clf_gnb_3, X, y, cv=5)
f1_scores_gnb = np.append(f1_scores_gnb, np.mean(f1_scores_gnb))
np.savetxt('./f1/p100-p10-no-encoding-gnb.txt', f1_scores_gnb, fmt='%1.3f')
