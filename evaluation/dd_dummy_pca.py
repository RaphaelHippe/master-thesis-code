import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import PermutationImportance
import category_encoders as ce
from sklearn.decomposition import PCA

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


df = pd.read_csv('./../datasets/damage_done.csv')
y = df['ct_wins']

# features = ['pc0','pc1','pc2','pc3','pc4','pc12','pc13','pc14','pc15','pc16','pc23','pc24','pc26']

X0 = df.drop(['ct_wins', 't_wins'], axis=1)
X0 = X0.rename(index=str, columns={
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

pcs = [20, 27, 30, 35, 40, 45]
results = []

for pc in pcs:
    X_dummy = pd.get_dummies(X0)
    pca = PCA(n_components=pc)
    X_dummy_pca = pca.fit_transform(X_dummy)
    X = pd.DataFrame(X_dummy_pca, columns=['pc{}'.format(i) for i in range(pc)])
    results.append(doAccuracyTests(X, y, 'pca-{}'.format(pc)))


df = pd.DataFrame(results, columns=['encoding', 'knn', 'rfc', 'gnb'])
df.to_csv('./acc/dd_cv50_dummy_pca.csv')
