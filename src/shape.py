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

D.compile(loss='mse', optimizer='rmsprop')
D.summary()
#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

epochs = 25

hist_update = D.fit_generator(
    DataGenerator(X_train, Y_train,
                  batch_size=128, data_augment=True, shape_learning=True), epochs=epochs,
    validation_data=DataGenerator(X_train, Y_train, batch_size=128,
                                  data_augment=False, shape_learning=True), validation_steps=100).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

D.save_weights("shape_weights.h5")
pickle.dump(history, open("shape_history.p", "wb"))