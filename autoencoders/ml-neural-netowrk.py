import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


columns = ['encoder', 'model', 'loss', 'acc']
results = []
COLOR_V = "10c"
NAME = "{}_150".format(COLOR_V)
scaler = MinMaxScaler()
df = pd.read_pickle('./../motivation-example/goodBadPlates_{}_test.pkl'.format(COLOR_V))

X_ns = df[['diameter', 'color']]
y = df['goodPlate']
y_binary = to_categorical(y)

X_s = X_ns.copy()
X_s[['diameter']] = scaler.fit_transform(X_s[['diameter']])

X_ns = pd.get_dummies(X_ns)
X_s = pd.get_dummies(X_s)

scaling = ['not_scaled', 'scaled']
activation_functions = ['relu', 'sigmoid', 'hard_sigmoid', 'tanh']
loss_functions = ['categorical_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity', 'mean_squared_error']

for S in scaling:
    for LOSS_FN in loss_functions:
        for ACTIVATION_FN in activation_functions:
            encoder = load_model('./encoder/encoder_{}_{}_{}_{}.h5'.format(NAME, S, LOSS_FN, ACTIVATION_FN))
            if S == 'scaled':
                X_encoded = encoder.predict(X_s)
            if S == 'not_scaled':
                X_encoded = encoder.predict(X_ns)

            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_binary, test_size=0.33, random_state=42)


            # ==============================
            # MODEL
            for LOSS_FN_2 in loss_functions:
                for ACTIVATION_FN_2 in activation_functions:
                    model = Sequential()
                    model.add(
                        Dense(2, input_shape=(2,), activation=ACTIVATION_FN_2)
                    )
                    model.add(
                        Dense(2, activation=ACTIVATION_FN_2)
                    )
                    # model.add(
                    #     Dense(4, activation=ACTIVATION_FN_2)
                    # )
                    model.add(
                        Dense(2, activation='hard_sigmoid')
                    )
                    print model.summary()
                    # tensorboard = TensorBoard(log_dir="./nn_logs/{}_{}_{}_{}_v2".format(NAME, LOSS_FN, ACTIVATION_FN, scaled))
                    model.compile(optimizer='adam', loss=LOSS_FN_2, metrics=['acc'])
                    model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    # callbacks=[tensorboard],
                    validation_data=(X_test, y_test))
                    # model.save('./nn/{}_{}_{}_{}_v2.h5'.format(NAME, LOSS_FN, ACTIVATION_FN, scaled))

                    model_metrics = model.evaluate(X_test, y_test)
                    results.append([
                        '{}_{}_{}_{}'.format(NAME, S, LOSS_FN, ACTIVATION_FN),
                        '{}_{}'.format(LOSS_FN_2, ACTIVATION_FN_2),
                        model_metrics[0],
                        model_metrics[1]
                    ])
                    model = None
            encoder = None



results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv('./ml-nn-encoded.csv')
