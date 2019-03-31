import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import PermutationImportance
import category_encoders as ce

df = pd.read_csv('./../datasets/damage_done.csv')
y = df['ct_wins']

nominal_columns = [
'weapon',
'attackerSpotted',
'attackerSide',
'attackerIsScoped',
'attackerIsDucked',
'attackerIsDucking',
'attackerHasHelmet',
'victimSide',
'victimIsDucked',
'victimIsDucking',
'victimIsDefusing',
'victimIsScoped',
'victimHasHelmet',
'hitgroup',
]
features = [
"x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15",
"x16","x17","x18","x19","x20","x21","x22","x23","x24","x25","x26","x27"
]

X0 = df.drop(['ct_wins', 't_wins'], axis=1)


encoder = ce.OrdinalEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)

X = X.rename(index=str, columns={
    "attackerHealth": "x1",
    "attackerXPosition": "x2",
    "attackerYPosition": "x3",
    "attackerZPosition": "x4",
    "weapon": "x5",
    "attackerSpotted": "x6",
    "attackerSide": "x7",
    "attackerIsScoped": "x8",
    "attackerIsDucked": "x9",
    "attackerIsDucking": "x10",
    "attackerHasHelmet": "x11",
    "victimRemainingHealth": "x12",
    "victimRemainingArmor": "x13",
    "victimXPosition": "x14",
    "victimYPosition": "x15",
    "victimZPosition": "x16",
    "victimSide": "x17",
    "victimIsDucked": "x18",
    "victimIsDucking": "x19",
    "victimIsDefusing": "x20",
    "victimIsScoped": "x21",
    "victimHasHelmet": "x22",
    "damageArmor": "x23",
    "damageHealth": "x24",
    "hitgroup": "x25",
    "roundTime": "x26",
    "round_number": "x27"
})


# Permutaiton
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(X_train, y_train)
    pi_knn = PermutationImportance.PermutationImportance(clf_knn)
    pi_knn.fit(X_test, y_test, features=features)
    pi_knn.store('dd-ordinal-knn')
    # KNN
    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train, y_train)
    pi_rfc = PermutationImportance.PermutationImportance(clf_rfc)
    pi_rfc.fit(X_test, y_test, features=features)
    pi_rfc.store('dd-ordinal-rfc')
    # GNB
    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train, y_train)
    pi_gnb = PermutationImportance.PermutationImportance(clf_gnb)
    pi_gnb.fit(X_test, y_test, features=features)
    pi_gnb.store('dd-ordinal-gnb')


# ACC SCORE
# KNN
clf_knn_2 = KNeighborsClassifier(n_neighbors=3)
acc_scores_knn = cross_val_score(clf_knn_2, X, y, cv=5)
acc_scores_knn = np.append(acc_scores_knn, np.mean(acc_scores_knn))
np.savetxt('./acc/dd-ordinal-knn.txt', acc_scores_knn, fmt='%1.3f')
# RFC
clf_rfc_2 = RandomForestClassifier()
acc_scores_rfc = cross_val_score(clf_rfc_2, X, y, cv=5)
acc_scores_rfc = np.append(acc_scores_rfc, np.mean(acc_scores_rfc))
np.savetxt('./acc/dd-ordinal-rfc.txt', acc_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_2 = GaussianNB()
acc_scores_gnb = cross_val_score(clf_gnb_2, X, y, cv=5)
acc_scores_gnb = np.append(acc_scores_gnb, np.mean(acc_scores_gnb))
np.savetxt('./acc/dd-ordinal-gnb.txt', acc_scores_gnb, fmt='%1.3f')


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
np.savetxt('./f1/dd-ordinal-knn.txt', f1_scores_knn, fmt='%1.3f')
# RFC
clf_rfc_3 = RandomForestClassifier()
f1_scores_rfc = cross_val_f1_score(clf_rfc_3, X, y, cv=5)
f1_scores_rfc = np.append(f1_scores_rfc, np.mean(f1_scores_rfc))
np.savetxt('./f1/dd-ordinal-rfc.txt', f1_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_3 = GaussianNB()
f1_scores_gnb = cross_val_f1_score(clf_gnb_3, X, y, cv=5)
f1_scores_gnb = np.append(f1_scores_gnb, np.mean(f1_scores_gnb))
np.savetxt('./f1/dd-ordinal-gnb.txt', f1_scores_gnb, fmt='%1.3f')
