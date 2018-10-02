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

inputs = Input(shape=(X_train.shape[1],))
Dx = Dense(128, activation="relu")(inputs)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(10, activation="relu")(Dx)
Dx = Dense(1, activation="linear")(Dx)
D = Model([inputs], [Dx], name='D')

D.compile(loss='mse', optimizer='rmsprop')

#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

batch_size = 128
epochs = 35

hist_update = D.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1).history

history.update([('loss',
                 history['loss'] + hist_update['loss']),
                ('val_loss',
                 history['val_loss'] + hist_update['val_loss'])])


D.save_weights("first_weights.h5")
pickle.dump(history, open("first_history.p", "wb"))
