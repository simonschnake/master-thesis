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

from keras import losses
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import h5py
import pickle
import numpy as np

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

history = {'loss': [], 'val_loss': []}

##############################################################
#                        _      _                            #
#        /\/\   ___   __| | ___| |                           #
#       /    \ / _ \ / _` |/ _ \ |                           #
#      / /\/\ \ (_) | (_| |  __/ |                           #
#      \/    \/\___/ \__,_|\___|_|                           #
##############################################################

lam = 0.5

inputs = Input(shape=(X_train.shape[1],))
Dx = Dense(128, activation="relu")(inputs)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(10, activation="relu")(Dx)
Dx = Dense(1, activation="linear")(Dx)
D = Model([inputs], [Dx])

results = Input(shape=(Y_train.shape[1],))
Rx = Lambda(lambda x: (x[0]-x[1])/x[1]**0.5)([D(inputs), results])
Rx = Dense(10, activation="relu")(Rx)
Rx = Dense(20, activation="relu")(Rx)
Rx = Dense(500, activation="softmax")(Rx)
R = Model([inputs, results], [Rx])


def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * losses.mean_squared_error(y_pred, y_true)
    return loss_D


def make_loss_R(c):
    def loss_R(z_true, z_pred):
        return c * losses.categorical_crossentropy(z_pred, z_true)
    return loss_R


opt = Adagrad()
D.compile(loss=[make_loss_D(c=1.0)], optimizer=opt)
D.trainable = True
R.trainable = False
DRf = Model([inputs, results], [D(inputs), R([inputs, results])])
DRf.compile(loss=[make_loss_D(c=1.0), make_loss_R(c=-lam)], optimizer=opt)
D.trainable = False
R.trainable = True
DfR = Model([inputs, results], [R([inputs, results])])
DfR.compile(loss=[make_loss_R(c=1.0)], optimizer=opt)

#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0.01,
                          patience=3,
                          verbose=0, mode='auto')
callbacks_list = [earlystop]

batch_size = 128
epochs = 5

D.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
      validation_split=0.1, callbacks=callbacks_list)

bins = np.arange(0., 10., 10./500.)[:-1]
Z_train = np.digitize(Y_train, bins=bins)
Z_train = np_utils.to_categorical(Z_train, num_classes=500)

for i in range(105):

    # Fit R
    d_weights = D.get_weights()

    DfR.fit([X_train, Y_train],
            Z_train,
            # epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks_list)

    print(d_weights[2][0] == D.get_weights()[2][0])

    # Fit D
    r_weights = R.get_weights()
    hist_update = DRf.fit([X_train, Y_train],
                          [Y_train, Z_train],
                          # epochs=epochs,
                          batch_size=batch_size,
                          validation_split=0.1,
                          callbacks=callbacks_list).history
    history.update([('loss',
                     history['loss'] + hist_update['loss']),
                    ('val_loss',
                     history['val_loss'] + hist_update['val_loss'])])
    r_weights = R.get_weights()

D.save_weights("adversarial_weights.h5")
pickle.dump(history, open("adversarial_history.p", "wb"))
