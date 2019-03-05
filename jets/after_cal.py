##############################################################
#    ___           _                                         #
#   / _ \__ _  ___| | ____ _  __ _  ___  ___                 #
#  / /_)/ _` |/ __| |/ / _` |/ _` |/ _ \/ __|                #
# / ___/ (_| | (__|   < (_| | (_| |  __/\__ \                #
# \/    \__,_|\___|_|\_\__,_|\__, |\___||___/                #
#                            |___/                           #
##############################################################

from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import Sequence
from keras import backend as K
import h5py
import numpy as np
import pickle
import tensorflow as tf

##############################################################
#  _                 _ _                  ___      _         #
# | | ___   __ _  __| (_)_ __   __ _     /   \__ _| |_ __ _  #
# | |/ _ \ / _` |/ _` | | '_ \ / _` |   / /\ / _` | __/ _` | #
# | | (_) | (_| | (_| | | | | | (_| |  / /_// (_| | || (_| | #
# |_|\___/ \__,_|\__,|__|_| |_|\__, | /___,' \__,_|\__\__,_| #
#                              |___/                         #
##############################################################


class DataGenerator(Sequence):

    def __init__(self, batch_size=128, train=True):
        self.batch_size = batch_size
        try:
            data = h5py.File('./QCD_Pt-30to150Run2Spring18.h5', 'r')
        except OSError:
            print('Data not found')
            exit()

        sess_str = "val"
        if train is True:
            sess_str = "train"
        self.y = data[sess_str + '_eventValues']

    def __len__(self):
        return int(np.floor(len(self.y) / float(self.batch_size))*0.1)

    def __getitem__(self, idx):
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_y[:, [3, 4, 5]], batch_y[:, 7]


history = {'loss': [], 'val_loss': []}

##############################################################
#                        _      _                            #
#        /\/\   ___   __| | ___| |                           #
#       /    \ / _ \ / _` |/ _ \ |                           #
#      / /\/\ \ (_) | (_| |  __/ |                           #
#      \/    \/\___/ \__,_|\___|_|                           #
##############################################################


def mean_squared_percentage_error(y_true, y_pred):
    diff = K.square((y_true - y_pred) / K.clip(K.abs(y_true),
                                               K.epsilon(),
                                               None))
    return 100. * K.mean(diff, axis=-1)


def accuracy(y_true, y_pred):
    R = K.abs(y_pred)/K.clip(K.abs(y_true),
                             K.epsilon(),
                             None)
    mu = K.abs(K.mean(R, axis=-1)-1.)
    sigma = K.std(R, axis=-1)
    return K.exp(-mu-sigma)


# network architecture parameters
inputs = Input(shape=(3,))
Dx = Dense(8, activation="relu")(inputs)
Dx = Dense(4, activation="relu")(Dx)
Dx = Dense(3, activation="relu")(Dx)
Dx = Dense(2, activation="relu")(Dx)
Dx = Dense(1, activation="linear")(Dx)
D = Model([inputs], [Dx], name='D')

D.compile(loss=mean_squared_percentage_error, optimizer='rmsprop', metrics=[accuracy])

#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

epochs = 1
train_Gen = DataGenerator(batch_size=128, train=True)
val_Gen = DataGenerator(batch_size=128, train=False)

hist_update = D.fit_generator(train_Gen,
                              epochs=epochs,
                              validation_data=val_Gen,
                              validation_steps=100).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

D.save_weights("aftercal_weights.h5")
pickle.dump(history, open("aftercal_history.p", "wb"))
