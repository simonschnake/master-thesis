# This code is a adaptation of the code
# https://github.com/glouppe/paper-learning-to-pivot
# compile every model after setting trainable flags again

##############################################################
#    ___           _                                         #
#   / _ \__ _  ___| | ____ _  __ _  ___  ___                 #
#  / /_)/ _` |/ __| |/ / _` |/ _` |/ _ \/ __|                #
# / ___/ (_| | (__|   < (_| | (_| |  __/\__ \                #
# \/    \__,_|\___|_|\_\__,_|\__, |\___||___/                #
#                            |___/                           #
##############################################################
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, Activation
from keras.models import Model
import h5py
import pickle

from utils import DataGenerator
##############################################################
#  _                 _ _                  ___      _         #
# | | ___   __ _  __| (_)_ __   __ _     /   \__ _| |_ __ _  #
# | |/ _ \ / _` |/ _` | | '_ \ / _` |   / /\ / _` | __/ _` | #
# | | (_) | (_| | (_| | | | | | (_| |  / /_// (_| | || (_| | #
# |_|\___/ \__,_|\__,|__|_| |_|\__, | /___,' \__,_|\__\__,_| #
#                              |___/                         #
##############################################################

try:
    data = h5py.File('../../data/electron.h5', 'r')
except OSError:
    try:
        data = h5py.File('../data/electron.h5', 'r')
    except OSError:
        print('Data not found')
        exit()
X_train = data['train']['X']
Y_train = data['train']['Y']

X_test = data['test']['X']
Y_test = data['test']['Y']

history = {'loss': [], 'val_loss': []}

##############################################################
#                        _      _                            #
#        /\/\   ___   __| | ___| |                           #
#       /    \ / _ \ / _` |/ _ \ |                           #
#      / /\/\ \ (_) | (_| |  __/ |                           #
#      \/    \/\___/ \__,_|\___|_|                           #
##############################################################

inputs = Input(shape=(8, 8, 17, 1))
Dx = Conv3D(32, (3, 3, 3), padding='same')(inputs)
Dx = Activation('relu')(Dx)
Dx = Conv3D(10, (3, 3, 3))(Dx)
Dx = Activation('relu')(Dx)
Dx = Conv3D(5, (5, 5, 5), strides = (1, 1, 1), name = 'conv')(Dx)

Dx = Flatten()(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dropout(0.25)(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(10, activation="relu")(Dx)
Dx = Dense(1, activation="linear")(Dx)
D = Model([inputs], [Dx], name='D')

D.summary()

def likelihood_loss(y_true, y_pred):
    epsilon = tf.constant(0.0000001)
    mu = y_pred
    sigma = 0.32*tf.sqrt(y_true)
    first_part = tf.divide(tf.square(mu - y_true),
                           2.*tf.square(sigma)+epsilon)
    a = tf.divide(10.-mu, tf.sqrt(2.)*sigma+epsilon)
    b = tf.divide(0.-mu, tf.sqrt(2.)*sigma+epsilon)
    penalty = tf.erf(a) - tf.erf(b)
    loss = first_part + tf.log(penalty+epsilon) + tf.log(tf.sqrt(2.*np.pi)*sigma+epsilon)
    return tf.reduce_mean(loss)

D.compile(loss=likelihood_loss, optimizer='adadelta')
D.load_weights('data_augment_conv_weights.h5')

#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

epochs = 25

hist_update = D.fit_generator(DataGenerator(X_train, Y_train,
                                            batch_size=128,
                                            data_augment=True),
                              epochs=epochs,
                              validation_data=DataGenerator(X_test,
                                                            Y_test, batch_size=1000,
                                                            data_augment=False)).history

history.update([('loss',
                 history['loss'] + hist_update['loss']),
                ('val_loss',
                 history['val_loss'] + hist_update['val_loss'])])


D.save_weights("likelihood__conv_weights.h5")
pickle.dump(history, open("likelihood_conv_history.p", "wb"))
