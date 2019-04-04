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

from keras.layers import Input, Dense, Conv3D, Flatten, BatchNormalization, Activation, AveragePooling3D, Dropout
from keras.models import Model
import h5py
import pickle
import tensorflow as tf
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

##################################################################
#  _                    ______                _   _              #
# | |                   |  ___|              | | (_)             #
# | |     ___  ___ ___  | |_ _   _ _ __   ___| |_ _  ___  _ __   #
# | |    / _ \/ __/ __| |  _| | | | '_ \ / __| __| |/ _ \| '_ \  #
# | |___| (_) \__ \__ \ | | | |_| | | | | (__| |_| | (_) | | | | #
# \_____/\___/|___/___/ \_|  \__,_|_| |_|\___|\__|_|\___/|_| |_| #
##################################################################
                                                              
def make_loss(c):
    def likelihood_loss(y_true, y_pred):
        y_fit = y_true[1]
        mu = y_true[0]
        y_pred = (y_pred+1)*y_fit

        epsilon = tf.constant(0.0000001)
        sigma = c*tf.sqrt(y_true)
        first_part = tf.divide(tf.square(mu - y_true),
                               2.*tf.square(sigma)+epsilon)
        a = tf.divide(10.-mu, tf.sqrt(2.)*sigma+epsilon)
        b = tf.divide(0.-mu, tf.sqrt(2.)*sigma+epsilon)
        penalty = tf.erf(a) - tf.erf(b)
        loss = first_part + tf.log(penalty+epsilon) + tf.log(tf.sqrt(2.*np.pi)*sigma+epsilon)
        return tf.reduce_mean(loss)
    return likelihood_loss

def weighted_loss(y_true, y_pred):
    y_fit = y_true[1]
    mu = y_true[0]
    y_pred = (y_pred+1)*y_fit

    epsilon = tf.constant(0.0000001)
    sigma = tf.sqrt(y_true)
    first_part = tf.divide(tf.square(mu - y_true),
                           2.*tf.square(sigma)+epsilon)
    return tf.reduce_mean(first_part)


#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

D.compile(loss=weighted_loss, optimizer='rmsprop')

epochs = 150

hist_update = D.fit_generator(
    DataGenerator(X_train, Y_train,
                  batch_size=128, data_augment=True, shape_learning=True), epochs=epochs,
    validation_data=DataGenerator(X_train, Y_train, batch_size=128,
                                  data_augment=False, shape_learning=True), validation_steps=100).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

y_pred = D.predict_generator(DataGenerator(X_test, Y_test, batch_size=128, data_augment=False))
y_pred = y_pred.reshape(len(y_pred), )
y_true = np.array(Y_test)[:len(y_pred)].reshape(len(y_pred), )
y, mu, sigma = sliced_statistics(y_true, y_pred, n)

c = np.mean(sigma)

D.compile(loss=make_loss(c), optimizer='rmsprop')

epochs = 50

hist_update = D.fit_generator(
    DataGenerator(X_train, Y_train,
                  batch_size=128, data_augment=True, shape_learning=True), epochs=epochs,
    validation_data=DataGenerator(X_train, Y_train, batch_size=128,
                                  data_augment=False, shape_learning=True), validation_steps=100).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

y_pred = D.predict_generator(DataGenerator(X_test, Y_test, batch_size=128, data_augment=False))
y_pred = y_pred.reshape(len(y_pred), )
y_true = np.array(Y_test)[:len(y_pred)].reshape(len(y_pred), )
y, mu, sigma = sliced_statistics(y_true, y_pred, n)


results = {'y_pred': y_pred,
           'y_true': y_true,
           'y': y,
           'mu': mu,
           'sigma': sigma,
           'history': history}

pickle.dump(history, open("shape_history.p", "wb"))
