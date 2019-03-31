import sys

import pandas as pd
import numpy as np

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

NAME = "10c_150_coloronly"

df = pd.read_pickle('./../motivation-example/goodBadPlates_10c_test.pkl')

X = df['color']
y = df['goodPlate']

X = pd.get_dummies(X)

X_train, X_test, _, _ = train_test_split(
    X, y, test_size=0.33, random_state=42)

input_dim = X_train.shape[1]
encoding_dims = [1,2]
activation_functions = ['relu', 'sigmoid', 'hard_sigmoid', 'tanh']
loss_functions = ['categorical_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity', 'mean_squared_error']
for encoding_dim in encoding_dims:
    for LOSS_FN in loss_functions:
        for ACTIVATION_FN in activation_functions:

            compression_factor = float(input_dim) / encoding_dim
            print "Compression factor: {}".format(compression_factor)

            # ===
            # Autoencoder
            # ===
            autoencoder = Sequential()
            autoencoder.add(
                Dense(8, input_shape=(input_dim,), activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(4, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(4, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(8, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(input_dim, activation='hard_sigmoid')
            )

            print autoencoder.summary()

            # ===
            # Encoder
            # ===
            input_data = Input(shape=(input_dim,))
            encoder_layer1 = autoencoder.layers[0]
            encoder_layer2 = autoencoder.layers[1]
            encoder_layer3 = autoencoder.layers[2]
            encoder = Model(input_data, encoder_layer3(encoder_layer2(encoder_layer1(input_data))))

            print encoder.summary()

            # ===
            # Fitting
            # ===
            tensorboard = TensorBoard(log_dir="./logs/{}_{}_{}_{}".format(NAME, LOSS_FN, ACTIVATION_FN, encoding_dim))
            autoencoder.compile(optimizer='adam', loss=LOSS_FN)
            # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
            autoencoder.fit(X_train, X_train,
            epochs=500,
            batch_size=256,
            callbacks=[tensorboard],
            validation_data=(X_test, X_test))


            autoencoder.save('./autoencoder/autoencoder_{}_{}_{}_{}.h5'.format(NAME, LOSS_FN, ACTIVATION_FN, encoding_dim))
            encoder.save('./encoder/encoder_{}_{}_{}_{}.h5'.format(NAME, LOSS_FN, ACTIVATION_FN, encoding_dim))

            # ===
            # Encode data
            # ===
            X_encoded = encoder.predict(X)

            # ===
            # plot encoded data
            # ===
            fig, ax = plt.subplots()

            data = pd.DataFrame(X_encoded).join(y)
            goodPlates = data.loc[data['goodPlate'] == True]
            badPlates = data.loc[data['goodPlate'] == False]

            if encoding_dim == 1:
                ax.scatter(goodPlates[0], [0 for i in xrange(len(goodPlates))], c='r')
                ax.scatter(badPlates[0], [0 for i in xrange(len(badPlates))], c='b')
            else:
                ax.scatter(goodPlates[0], goodPlates[1], c='r')
                ax.scatter(badPlates[0], badPlates[1], c='b')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Motivation example: {} {} {} {}'.format(NAME, LOSS_FN, ACTIVATION_FN, encoding_dim))
            # plt.show()
            fig.savefig("./images/10c_only/good_bad_plates_autoencoded_dimensions_{}_{}_{}_{}.png".format(NAME, LOSS_FN, ACTIVATION_FN, encoding_dim))
            plt.close('all')
