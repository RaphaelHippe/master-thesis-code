import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import PermutationImportance
import category_encoders as ce

df = pd.read_csv('./../datasets/plates_10.csv')
y = df['goodPlate']
X = df.drop(['goodPlate', 'Unnamed: 0', 'color'], axis=1)
# ACC SCORE
# KNN
clf_knn_2 = KNeighborsClassifier(n_neighbors=3)
acc_scores_knn = cross_val_score(clf_knn_2, X, y, cv=50)
acc_scores_knn = np.append(acc_scores_knn, np.mean(acc_scores_knn))
np.savetxt('./acc/p10-diameter-knn-50.txt', acc_scores_knn, fmt='%1.3f')
# RFC
clf_rfc_2 = RandomForestClassifier()
acc_scores_rfc = cross_val_score(clf_rfc_2, X, y, cv=50)
acc_scores_rfc = np.append(acc_scores_rfc, np.mean(acc_scores_rfc))
np.savetxt('./acc/p10-diameter-rfc-50.txt', acc_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_2 = GaussianNB()
acc_scores_gnb = cross_val_score(clf_gnb_2, X, y, cv=50)
acc_scores_gnb = np.append(acc_scores_gnb, np.mean(acc_scores_gnb))
np.savetxt('./acc/p10-diameter-gnb-50.txt', acc_scores_gnb, fmt='%1.3f')


# F1 score
from sklearn.metrics import f1_score

def cross_val_f1_score(clf, X, y, cv=50):
    scores = []
    for _ in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(f1_score(y_test, y_pred, average='micro'))
    return scores

# KNN
clf_knn_3 = KNeighborsClassifier(n_neighbors=3)
f1_scores_knn = cross_val_f1_score(clf_knn_3, X, y, cv=50)
f1_scores_knn = np.append(f1_scores_knn, np.mean(f1_scores_knn))
np.savetxt('./f1/p10-diameter-knn-50.txt', f1_scores_knn, fmt='%1.3f')
# RFC
clf_rfc_3 = RandomForestClassifier()
f1_scores_rfc = cross_val_f1_score(clf_rfc_3, X, y, cv=50)
f1_scores_rfc = np.append(f1_scores_rfc, np.mean(f1_scores_rfc))
np.savetxt('./f1/p10-diameter-rfc-50.txt', f1_scores_rfc, fmt='%1.3f')
# GNB
clf_gnb_3 = GaussianNB()
f1_scores_gnb = cross_val_f1_score(clf_gnb_3, X, y, cv=50)
f1_scores_gnb = np.append(f1_scores_gnb, np.mean(f1_scores_gnb))
np.savetxt('./f1/p10-diameter-gnb-50.txt', f1_scores_gnb, fmt='%1.3f')
