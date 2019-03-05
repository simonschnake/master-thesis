##############################################################
#    ___           _                                         #
#   / _ \__ _  ___| | ____ _  __ _  ___  ___                 #
#  / /_)/ _` |/ __| |/ / _` |/ _` |/ _ \/ __|                #
# / ___/ (_| | (__|   < (_| | (_| |  __/\__ \                #
# \/    \__,_|\___|_|\_\__,_|\__, |\___||___/                #
#                            |___/                           #
##############################################################

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.utils import Sequence
from keras import backend as K
import h5py
import numpy as np
import pickle

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
        self.x = data[sess_str + '_pfCanValues']

    def __len__(self):
        return int(np.floor(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y[:, 7]


history = {'loss': [], 'val_loss': []}

##############################################################
#                        _      _                            #
#        /\/\   ___   __| | ___| |                           #
#       /    \ / _ \ / _` |/ _ \ |                           #
#      / /\/\ \ (_) | (_| |  __/ |                           #
#      \/    \/\___/ \__,_|\___|_|                           #
##############################################################

inputs = Input(shape=(200, 4,))
Dx = Flatten()(inputs)
Dx = Dense(800, activation="relu")(Dx)
Dx = Dense(700, activation="relu")(Dx)
Dx = Dense(600, activation="relu")(Dx)
Dx = Dense(500, activation="relu")(Dx)
Dx = Dense(400, activation="relu")(Dx)
Dx = Dense(300, activation="relu")(Dx)
Dx = Dense(200, activation="relu")(Dx)
Dx = Dense(1, activation="linear")(Dx)
D = Model([inputs], [Dx], name='D')


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
                              validation_steps=len(val_Gen)).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

D.save_weights("first_weights.h5")
pickle.dump(history, open("first_history.p", "wb"))
