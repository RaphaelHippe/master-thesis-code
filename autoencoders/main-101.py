import pandas as pd
import numpy as np

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

NAME = "100c_150"

scaler = MinMaxScaler()

# df = pd.read_pickle('./goodBadPlates.pkl')
# df = pd.read_pickle('./../motivation-example/goodBadPlates_100_color_500_000_plates.pkl')
df = pd.read_pickle('./../motivation-example/goodBadPlates_100c_test.pkl')

X_ns = df[['diameter', 'color']]
y = df['goodPlate']

X_s = X_ns
X_s[['diameter']] = scaler.fit_transform(X_s[['diameter']])

X_ns = pd.get_dummies(X_ns)
X_s = pd.get_dummies(X_s)

X_train_ns, X_test_ns, _, _ = train_test_split(
    X_ns, y, test_size=0.33, random_state=42)

X_train_s, X_test_s, _, _ = train_test_split(
    X_s, y, test_size=0.33, random_state=42)

if len(X_train_ns) != len(X_train_s):
    sys.exit("X_train_ns != X_train_s")

input_dim = X_train_ns.shape[1]
encoding_dim = 2

scaling = ['scaled', 'not_scaled']
activation_functions = ['relu', 'sigmoid', 'hard_sigmoid', 'tanh']
loss_functions = ['categorical_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity', 'mean_squared_error']
for S in scaling:
    for LOSS_FN in loss_functions:
        for ACTIVATION_FN in activation_functions:

            compression_factor = float(input_dim) / encoding_dim
            print "Compression factor: {}".format(compression_factor)

            # ===
            # Autoencoder
            # ===
            autoencoder = Sequential()
            autoencoder.add(
                Dense(32 * encoding_dim, input_shape=(input_dim,), activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(16 * encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(8 * encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(4 * encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(2 * encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(2 * encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(4 * encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(8 * encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(16 * encoding_dim, activation=ACTIVATION_FN)
            )
            autoencoder.add(
                Dense(32 * encoding_dim, activation=ACTIVATION_FN)
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
            encoder_layer4 = autoencoder.layers[3]
            encoder_layer5 = autoencoder.layers[4]
            encoder = Model(input_data,
                encoder_layer5(
                encoder_layer4(
                encoder_layer3(
                encoder_layer2(
                encoder_layer1(input_data))))))

            print encoder.summary()

            # ===
            # Fitting
            # ===
            tensorboard = TensorBoard(log_dir="./logs/{}_{}_{}_{}".format(NAME, S, LOSS_FN, ACTIVATION_FN))
            autoencoder.compile(optimizer='adam', loss=LOSS_FN)
            if S == 'scaled':
                autoencoder.fit(X_train_s, X_train_s,
                epochs=500,
                batch_size=256,
                callbacks=[tensorboard],
                validation_data=(X_test_s, X_test_s))
            if S == 'not_scaled':
                autoencoder.fit(X_train_ns, X_train_ns,
                epochs=500,
                batch_size=256,
                callbacks=[tensorboard],
                validation_data=(X_test_ns, X_test_ns))


            autoencoder.save('./autoencoder/autoencoder_{}_{}_{}_{}.h5'.format(NAME, S, LOSS_FN, ACTIVATION_FN))
            encoder.save('./encoder/encoder_{}_{}_{}_{}.h5'.format(NAME, S, LOSS_FN, ACTIVATION_FN))

            # ===
            # Encode data
            # ===
            if S == 'scaled':
                X_encoded = encoder.predict(X_s)
            if S == 'not_scaled':
                X_encoded = encoder.predict(X_ns)

            # ===
            # plot encoded data
            # ===
            fig, ax = plt.subplots()

            data = pd.DataFrame(X_encoded).join(y)
            goodPlates = data.loc[data['goodPlate'] == True]
            badPlates = data.loc[data['goodPlate'] == False]

            ax.scatter(goodPlates[0], goodPlates[1], c='r')
            ax.scatter(badPlates[0], badPlates[1], c='b')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Motivation example: {} {} {} {}'.format(NAME, S, LOSS_FN, ACTIVATION_FN))
            # plt.show()
            fig.savefig("./images/100c/good_bad_plates_autoencoded_dimensions_{}_{}_{}_{}.png".format(NAME, S, LOSS_FN, ACTIVATION_FN))
            plt.close('all')
