import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import PermutationImportance
import category_encoders as ce

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

df = pd.read_csv('./../nom2num/dd_nom2num.csv')
y = df['ct_wins']
X = df.drop(['ct_wins', 'Unnamed: 0'], axis=1)
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

results = []
results.append(doAccuracyTests(X, y, 'nom2num'))

df = pd.DataFrame(results, columns=['encoding', 'knn', 'rfc', 'gnb'])
df.to_csv('./acc/dd_cv50_nom2num.csv')
