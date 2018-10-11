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
from keras.layers import Input, Dense, Conv2D, Flatten, Activation, Lambda
from keras.models import Model
from keras.optimizers import Adadelta
import h5py
import pickle
from utils import DataGenerator
import numpy as np
from keras.utils import np_utils

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

history = {'D_loss': [], 'val_D_loss': [], 'R_loss': [], 'val_R_loss': []}

##############################################################
#                        _      _                            #
#        /\/\   ___   __| | ___| |                           #
#       /    \ / _ \ / _` |/ _ \ |                           #
#      / /\/\ \ (_) | (_| |  __/ |                           #
#      \/    \/\___/ \__,_|\___|_|                           #
##############################################################

lam = 0.5

inputs = Input(shape=(8, 8, 17,))
Dx = Conv2D(32, (2, 2), strides=(1, 1))(inputs)
Dx = Activation('relu')(Dx)
Dx = Flatten()(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(10, activation="relu")(Dx)
Dx = Dense(1, activation="linear")(Dx)
D = Model([inputs], [Dx], name='D')


cases = 500
results = Input(shape=(Y_train.shape[1],))
Rx = Lambda(lambda x: (x[0]-x[1])/x[1]**0.5)([D(inputs), results])
# Rx = Lambda(lambda x: (x[0]-x[1]))([D(inputs), results])
Rx = Dense(10, activation="relu")(Rx)
Rx = Dense(10, activation="relu")(Rx)
Rx = Dense(cases, activation="softmax")(Rx)
R = Model([inputs, results], [Rx], name='R')


def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * losses.mean_squared_error(y_pred, y_true)
    return loss_D


def make_loss_R(c):
    def loss_R(z_true, z_pred):
        return c * losses.categorical_crossentropy(z_pred, z_true)
    return loss_R


opt = Adadelta()
D.compile(loss=[make_loss_D(c=1.0)], optimizer=opt)
D.load_weights('data_augment_weights.h5')

R.trainable = False
D.trainable = True
DRf = Model([inputs, results], [D(inputs), R([inputs, results])])
DRf.compile(loss=[make_loss_D(c=1.0), make_loss_R(c=-lam)], optimizer=opt)
R.trainable = True
D.trainable = False
DfR = Model([inputs, results], [R([inputs, results])])
DfR.compile(loss=[make_loss_R(c=lam)], optimizer=opt)

DfR.summary()
DRf.summary()

#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

bins = np.arange(0., 10., 10./cases)[:-1]
Z_train = np.digitize(Y_train, bins=bins)
Z_train = np_utils.to_categorical(Z_train, num_classes=cases)

for i in range(7):

    # Fit R
    dw = D.get_weights()
    rw = R.get_weights()
    hist_update = DfR.fit_generator(
        DataGenerator(X_train, Y_train, Z_train, adversary=True),
        epochs=3,
        validation_data=DataGenerator(X_train, Y_train, Z_train,
                                      data_augment=True,
                                      adversary=True)).history
    history.update([
        ('R_loss', history['R_loss'] + hist_update['loss']),
        ('val_R_loss', history['val_R_loss'] + hist_update['val_loss'])])
    
    # Fit D
    hist_update = DRf.fit_generator(
        DataGenerator(X_train, Y_train, Z_train,
                      adversary=False),
        epochs=3,
        validation_data=DataGenerator(
            X_train, Y_train,
            data_augment=False,
            adv_cases=cases,
            adversary=False)).history

    history.update([
        ('D_loss', history['D_loss'] + hist_update['D_loss']),
        ('val_D_loss', history['val_D_loss'] + hist_update['val_D_loss']),
        ('R_loss', history['R_loss'] + [-x for x in hist_update['R_loss']]),
        ('val_R_loss', history['val_R_loss']
         + [-x for x in hist_update['val_R_loss']])])

D.save_weights("adversarial_weights.h5")
pickle.dump(history, open("adversarial_history.p", "wb"))
