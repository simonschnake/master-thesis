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
d_result = Input(shape=(Y_train.shape[1],))
results = Input(shape=(Y_train.shape[1],))
Rx = Lambda(lambda x: (x[0]-x[1])/x[1]**0.5)([d_result, results])
Rx = Dense(10, activation="relu")(Rx)
Rx = Dense(20, activation="relu")(Rx)
Rx = Dense(cases, activation="softmax")(Rx)
R = Model([d_result, results], [Rx], name='R')


def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * losses.mean_squared_error(y_pred, y_true)
    return loss_D


def make_loss_R(c):
    def loss_R(z_true, z_pred):
        return c * losses.categorical_crossentropy(z_pred, z_true)
    return loss_R


opt = Adadelta()

D.load_weights('data_augment_weights.h5')

R.trainable = False
D.trainable = True
train_D = Model([inputs, results], [D(inputs), R([D(inputs), results])])
train_D.compile(loss=[make_loss_D(c=1.0), make_loss_R(c=-lam)], optimizer='rmsprop')

R.trainable = True
D.trainable = False
train_R = Model([inputs, results], [R([D(inputs), results])])
train_R.compile(loss=[make_loss_R(c=lam)], optimizer='rmsprop')

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

pre_train_epochs = 2
epochs = 5
dg = DataGenerator(X_train, Y_train, Z_train, batch_size=256)
dg_it = iter(dg)

# for i in range(pre_train_epochs*len(dg)):
#     x, y, z = next(dg_it)
#     train_R.train_on_batch([x, y], z)
#     if (i % 100) is 0:
#         print("{0:.1f}%".format(100*i/(pre_train_epochs*len(dg))))

# for i in range(epochs*len(dg)):
#     x, y, z = next(dg_it)
#     train_R.train_on_batch([x, y], z)
#     train_D.train_on_batch([x, y], [y, z])
#     if (i % 100) is 0:
#         print("{0:.1f}%".format(100*i/(epochs*len(dg))))
# train_R.fit_generator(
#     DataGenerator(X_train, Y_train, Z_train,
#                   batch_size=256, adv=True, data_augment=False),
#     epochs=100)

for i in range(200):
    print(i)
    train_R.fit_generator(
        DataGenerator(X_train, Y_train, Z_train, batch_size=256, adv=True),
        epochs=5)
    train_D.fit_generator(
        DataGenerator(X_train, Y_train, Z_train, batch_size=256, adv=False),
        epochs=5)

D.save_weights("adversarial_weights.h5")
