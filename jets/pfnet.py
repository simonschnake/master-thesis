##############################################################
#    ___           _                                         #
#   / _ \__ _  ___| | ____ _  __ _  ___  ___                 #
#  / /_)/ _` |/ __| |/ / _` |/ _` |/ _ \/ __|                #
# / ___/ (_| | (__|   < (_| | (_| |  __/\__ \                #
# \/    \__,_|\___|_|\_\__,_|\__, |\___||___/                #
#                            |___/                           #
##############################################################

from keras.utils import Sequence
from keras.optimizers import adam
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

try:
    data = h5py.File('./QCD_Pt-30to150Run2Spring18.h5', 'r')
except OSError:
    print('Data not found')
    exit()


class DataGenerator(Sequence):

    def __init__(self, batch_size=128, train=True):
        self.batch_size = batch_size
        
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


history = {'val_loss': [], 'val_accuracy': [], 'val_mean_pred': [],
           'loss': [], 'accuracy': [], 'mean_pred': []}

##############################################################
#                        _      _                            #
#        /\/\   ___   __| | ___| |                           #
#       /    \ / _ \ / _` |/ _ \ |                           #
#      / /\/\ \ (_) | (_| |  __/ |                           #
#      \/    \/\___/ \__,_|\___|_|                           #
##############################################################


def binned_mean_squared_error(y_true, y_pred):
    # element of mean squared error
    squared_error = tf.square(y_pred - y_true)/y_true
    value_range = [30., 150.]
    # binning in y_true
    indices = tf.histogram_fixed_width_bins(y_true, value_range, nbins=50)
    # build mean per bin
    means_per_bin = tf.math.unsorted_segment_mean(squared_error, indices, 50)
    return tf.reduce_mean(means_per_bin, axis=None, keepdims=False)


def accuracy(y_true, y_pred):
    R = K.abs(y_pred)/K.clip(K.abs(y_true),
                             K.epsilon(),
                             None)
    mu = K.abs(K.mean(R, axis=-1)-1.)
    sigma = K.std(R, axis=-1)
    return K.exp(-mu-sigma)


def mean_pred(y_true, y_pred):
    R = K.abs(y_pred)/K.clip(K.abs(y_true),
                             K.epsilon(),
                             None)
    return K.mean(R)


# network architecture parameters
ppm_sizes = (50, 100, 128)
dense_sizes = (200, 100, 100, 50)


def binned_max_likelihood_loss(y_true, y_pred):
    epsilon = tf.constant(0.0000001)
    mu = y_pred
    sigma = 0.9*tf.sqrt(y_true)
    first_part = tf.divide(tf.square(mu - y_true),
                           2.*tf.square(sigma)+epsilon)
    a = tf.divide(mu-30., tf.sqrt(2.)*sigma+epsilon)
    b = tf.divide(mu-150., tf.sqrt(2.)*sigma+epsilon)
    penalty = tf.erf(a) - tf.erf(b)
    loss = first_part + tf.log(penalty+epsilon) + tf.log(sigma+epsilon)
    value_range = [30., 150.]
    # binning in y_true
    indices = tf.histogram_fixed_width_bins(y_true, value_range, nbins=50)
    # build mean per bin
    loss_per_bin = tf.math.unsorted_segment_mean(loss, indices, 50)
    return tf.reduce_mean(loss_per_bin, axis=None, keepdims=False)


def likelihood_loss(y_true, y_pred):
    epsilon = tf.constant(0.0000001)
    mu = y_pred
    sigma = 0.815*tf.sqrt(y_true)
    first_part = tf.divide(tf.square(mu - y_true),
                           2.*tf.square(sigma)+epsilon)
    a = tf.divide(150.-mu, tf.sqrt(2.)*sigma+epsilon)
    b = tf.divide(30.-mu, tf.sqrt(2.)*sigma+epsilon)
    penalty = tf.erf(a) - tf.erf(b)
    loss = first_part + tf.log(penalty+epsilon) + tf.log(sigma+epsilon)
    return tf.reduce_mean(loss, axis=None, keepdims=False)


pfn = PFN(input_dim=4,
          ppm_sizes=ppm_sizes,
          dense_sizes=dense_sizes,
          output_dim=1,
          output_act='linear',
          loss=binned_max_likelihood_loss,
          metrics=[accuracy, mean_pred],
          opt=adam)

# pfn.model.load_weights("pfn_weights.h5")


#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

epochs = 30
train_Gen = DataGenerator(batch_size=256, train=True)
val_Gen = DataGenerator(batch_size=256, train=False)

hist_update = pfn.model.fit_generator(train_Gen,
                                      # use_multiprocessing=True,
                                      # workers=3,
                                      max_queue_size=10,
                                      epochs=epochs,
                                      validation_data=val_Gen,
                                      validation_steps=len(val_Gen)).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] + hist_update['val_loss']),
                ('accuracy', history['accuracy'] + hist_update['accuracy']),
                ('val_accuracy', history['val_accuracy'] + hist_update['val_accuracy']),
                ('mean_pred', history['mean_pred'] + hist_update['mean_pred']),
                ('val_mean_pred', history['val_mean_pred'] + hist_update['val_mean_pred'])])


pfn.model.save_weights("pfn_weights.h5")
pickle.dump(history, open("pfn_history.p", "wb"))
