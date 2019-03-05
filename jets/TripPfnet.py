##############################################################
#    ___           _                                         #
#   / _ \__ _  ___| | ____ _  __ _  ___  ___                 #
#  / /_)/ _` |/ __| |/ / _` |/ _` |/ _ \/ __|                #
# / ___/ (_| | (__|   < (_| | (_| |  __/\__ \                #
# \/    \__,_|\___|_|\_\__,_|\__, |\___||___/                #
#                            |___/                           #
##############################################################

from keras.utils import Sequence
from keras.optimizers import rmsprop, adadelta, SGD
from keras import backend as K
import h5py
import numpy as np
import pickle
from energyflow.archs import PFN
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


def mean_squared_percentage_error(y_true, y_pred):
    diff = K.square(y_true - y_pred) / K.clip(K.abs(y_true),
                                              K.epsilon(),
                                              None)
    return 100. * K.mean(diff, axis=-1)

def accuracy(y_true, y_pred):
    R = K.abs(y_pred)/K.clip(K.abs(y_true),
                             K.epsilon(),
                             None)
    mu = K.abs(K.mean(R, axis=-1)-1.)
    sigma = K.std(R, axis=-1)
    return K.exp(-mu-sigma)


def R_min(y_true, y_pred):
    R = K.abs(y_pred)/K.clip(K.abs(y_true),
                             K.epsilon(),
                             None)
    return K.abs(K.mean(K.abs(R-1.), axis=-1))


def mean_pred(y_true, y_pred):
    R = K.abs(y_pred)/K.clip(K.abs(y_true),
                             K.epsilon(),
                             None)
    return K.mean(R)


# network architecture parameters
ppm_sizes = (100, 200, 200, 500)
dense_sizes = (200, 100, 100, 50)


def likelihood_loss(y_true, y_pred):
    epsilon = tf.constant(0.0000001)
    mu = y_pred
    sigma = 0.9*tf.sqrt(y_true)
    first_part = tf.divide(tf.square(mu - y_true),
                           2.*tf.square(sigma)+epsilon)
    a = tf.divide(10.-mu, tf.sqrt(2.)*sigma+epsilon)
    b = tf.divide(0.-mu, tf.sqrt(2.)*sigma+epsilon)
    penalty = tf.erf(a) - tf.erf(b)
    loss = first_part + tf.log(penalty+epsilon) + tf.log(tf.sqrt(2.*np.pi)*sigma+epsilon)
    return tf.reduce_mean(loss)


pfn = PFN(input_dim=4,
          ppm_sizes=ppm_sizes,
          dense_sizes=dense_sizes,
          output_dim=1,
          output_act='linear',
          loss=mean_squared_percentage_error,
          metrics=[accuracy, mean_pred],
          opt=rmsprop)

# pfn.model.load_weights("pfn_weights.h5")


#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

epochs = 10
train_Gen = DataGenerator(batch_size=128, train=True)
val_Gen = DataGenerator(batch_size=128, train=False)

hist_update = pfn.model.fit_generator(train_Gen,
                                      epochs=epochs,
                                      validation_data=val_Gen,
                                      validation_steps=100).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

pfn.model.save_weights("pfn_weights.h5")
pickle.dump(history, open("pfn_history.p", "wb"))
