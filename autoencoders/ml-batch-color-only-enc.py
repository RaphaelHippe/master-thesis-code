import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

results = []
COLOR_V = "100c"
NAME = "{}_150_coloronly".format(COLOR_V)
scaler = MinMaxScaler()
# df = pd.read_pickle('./../motivation-example/goodBadPlates_{}_test.pkl'.format(COLOR_V))
df = pd.read_csv('./../motivation-example/plates_100.csv')

X_color = df['color']
X_color = pd.get_dummies(X_color)

X_ns = df[['diameter', 'color']]
X_ns = pd.get_dummies(X_ns)
X_s = X_ns.copy()
X_s[['diameter']] = scaler.fit_transform(X_s[['diameter']])

y = df['goodPlate']

dimensions = [1,2]
scaling = ['not_scaled', 'scaled']
activation_functions = ['relu', 'sigmoid', 'hard_sigmoid', 'tanh']
loss_functions = ['categorical_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity', 'mean_squared_error']
for S in scaling:
    for D in dimensions:
        for LOSS_FN in loss_functions:
            for ACTIVATION_FN in activation_functions:
                encoder = load_model('./encoder/encoder_{}_{}_{}_{}.h5'.format(NAME, LOSS_FN, ACTIVATION_FN, D))
                X_encoded = encoder.predict(X_color)
                X = pd.DataFrame(X_encoded)
                if S == 'scaled':
                    X['diameter'] = X_s['diameter']
                else:
                    X['diameter'] = X_ns['diameter']

                # print X_encoded[0]
                # df = pd.DataFrame(X_encoded, columns=['a1', 'a2'])
                df = pd.DataFrame(X_encoded, columns=['a1', 'a2', 'a3', 'a4'])
                X['y'] = y
                X.to_csv('./encoded_datasets/p100_1/p100_1_{}_{}_{}_{}.csv'.format(S, LOSS_FN, ACTIVATION_FN, D))
#                 X_train, X_test, y_train, y_test = train_test_split(
#                     X, y, test_size=0.33, random_state=42)
#
#                 print D, S, LOSS_FN, ACTIVATION_FN
#                 clf1 = KNeighborsClassifier(n_neighbors=3)
#                 clf1.fit(X_train, y_train)
#                 score1 = clf1.score(X_test, y_test)
#                 print 'KNN', score1
#                 clf2 = SVC(gamma='auto')
#                 clf2.fit(X_train, y_train)
#                 score2 = clf2.score(X_test, y_test)
#                 print 'SVM', score2
#                 clf3 = RandomForestClassifier()
#                 clf3.fit(X_train, y_train)
#                 score3 = clf3.score(X_test, y_test)
#                 print 'RF', score3
#
#                 key = "{}_{}_{}_{}".format(S, LOSS_FN, ACTIVATION_FN, D)
#                 results.append({
#                 'key': key,
#                 'knn': score1,
#                 'svm': score2,
#                 'rf': score3
#                 })
#
# colors = np.empty((80,), dtype="S10")
# colors[0::3] = 'red'
# colors[1::3] = 'blue'
# colors[2::3] = 'green'
#
# fig, ax = plt.subplots()
#
# bar_x = []
# bar_height = []
# bar_x_labels = []
# bar_x_ticks = []
#
# counter = 1
# for res in results:
#     bar_x_labels.append(res['key'])
#     bar_x_ticks.append(counter)
#
#     bar_x.append(counter)
#     bar_height.append(res['knn'])
#     counter += 1
#     bar_x.append(counter)
#     bar_height.append(res['svm'])
#     counter += 1
#     bar_x.append(counter)
#     bar_height.append(res['rf'])
#     counter += 1
#
#
# line_x = [0]+bar_x+[len(bar_x)]
# # COINFLIP
# random_guess = np.array([0.5 for i in xrange(len(line_x))])
# ax.plot(line_x, random_guess, 'k-')
#
# # NOT ENCODED LEARNING
# X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(
#     X_ns, y, test_size=0.33, random_state=42)
# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
#     X_s, y, test_size=0.33, random_state=42)
#
# knn_ns = KNeighborsClassifier(n_neighbors=3)
# knn_s = KNeighborsClassifier(n_neighbors=3)
# svm_ns = SVC(gamma='auto')
# svm_s = SVC(gamma='auto')
# rf_ns = RandomForestClassifier()
# rf_s = RandomForestClassifier()
#
# knn_ns.fit(X_train_ns, y_train_ns)
# knn_s.fit(X_train_s, y_train_s)
# score_knn_ns = knn_ns.score(X_test_ns, y_test_ns)
# score_knn_s = knn_s.score(X_test_s, y_test_s)
# print score_knn_ns, score_knn_s
#
# svm_ns.fit(X_train_ns, y_train_ns)
# svm_s.fit(X_train_s, y_train_s)
# score_svm_ns = svm_ns.score(X_test_ns, y_test_ns)
# score_svm_s = svm_s.score(X_test_s, y_test_s)
# print score_svm_ns, score_svm_s
#
# rf_ns.fit(X_train_ns, y_train_ns)
# rf_s.fit(X_train_s, y_train_s)
# score_rf_ns = rf_ns.score(X_test_ns, y_test_ns)
# score_rf_s = rf_s.score(X_test_s, y_test_s)
# print score_rf_ns, score_rf_s
#
# # KNN NOT SCALED
# knn_acc1 = np.array([score_knn_ns for i in xrange(len(bar_x)/2)])
# ax.plot([i for i in xrange(len(bar_x)/2)], knn_acc1, 'r-')
# # KNN SCALED
# knn_acc2 = np.array([score_knn_s for i in xrange(len(bar_x)/2)])
# ax.plot([i for i in xrange(120, len(bar_x))], knn_acc2, 'r-')
# # SVM NOT SCALED
# svm_acc1 = np.array([score_svm_ns for i in xrange(len(bar_x)/2)])
# ax.plot([i for i in xrange(len(bar_x)/2)], svm_acc1, 'b-')
# # SVM SCALED
# svm_acc2 = np.array([score_svm_s for i in xrange(len(bar_x)/2)])
# ax.plot([i for i in xrange(120, len(bar_x))], svm_acc2, 'b-')
# # RF NOT SCALED
# rf_acc1 = np.array([score_rf_ns for i in xrange(len(bar_x)/2)])
# ax.plot([i for i in xrange(len(bar_x)/2)], rf_acc1, 'g-')
# # RF SCALED
# rf_acc2 = np.array([score_rf_s for i in xrange(len(bar_x)/2)])
# ax.plot([i for i in xrange(120, len(bar_x))], rf_acc2, 'g-')
#
# ax.plot([], [], 'o', c='red', label='KNN')
# ax.plot([], [], 'o', c='blue', label='SVM')
# ax.plot([], [], 'o', c='green', label='RF')
#
# ax.legend(loc='upper left')
# ax.set_xticks(bar_x_ticks)
# ax.set_xticklabels(bar_x_labels, minor=False)
# ax.bar(bar_x, bar_height, color=colors)
# ax.set_title('Motivation example ({}): Comparison of autoencoders (color only) for KNN and SVM accuracy'.format(COLOR_V))
# plt.xticks(rotation=90)
# # plt.tight_layout(pad=.01)
# plt.gcf().subplots_adjust(bottom=0.5)
# plt.show()
# fig.savefig("./rf-results-{}-color-only.png".format(COLOR_V))
