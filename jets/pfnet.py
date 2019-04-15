##############################################################
#    ___           _                                         #
#   / _ \__ _  ___| | ____ _  __ _  ___  ___                 #
#  / /_)/ _` |/ __| |/ / _` |/ _` |/ _ \/ __|                #
# / ___/ (_| | (__|   < (_| | (_| |  __/\__ \                #
# \/    \__,_|\___|_|\_\__,_|\__, |\___||___/                #
#                            |___/                           #
##############################################################

from keras.utils import Sequence
from keras.optimizers import RMSprop
from keras import backend as K
import h5py
import numpy as np
import pickle
from energyflow.archs import PFN
import tensorflow as tf
from scipy.stats import binned_statistic
from scipy.optimize import leastsq


##############################################################
#  _                 _ _                  ___      _         #
# | | ___   __ _  __| (_)_ __   __ _     /   \__ _| |_ __ _  #
# | |/ _ \ / _` |/ _` | | '_ \ / _` |   / /\ / _` | __/ _` | #
# | | (_) | (_| | (_| | | | | | (_| |  / /_// (_| | || (_| | #
# |_|\___/ \__,_|\__,|__|_| |_|\__, | /___,' \__,_|\__\__,_| #
#                              |___/                         #
##############################################################

class DataGenerator(Sequence):

    def __init__(self, batch_size=128, train=True, percent=1.0):
        self.batch_size = batch_size
        try:
            data = h5py.File('../../data/jets/QCD_Pt-30to150Run2Spring18.h5', 'r')
        except OSError:
            print('Data not found')
            exit()
        sess_str = "val"
        if train is True:
            sess_str = "train"
        self.y = data[sess_str + '_eventValues']
        self.x = data[sess_str + '_pfCanValues']
        self.percent = percent

    def __len__(self):
        return int(np.floor(len(self.x)*self.percent / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y[:, 7]
    def dump_res(self):
        return self.y[:, 7][:len(self)*self.batch_size]


history = {'loss': [], 'val_loss': []}


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
ppm_sizes = (100, 100, 128)
dense_sizes = (100, 100, 100)

pfn = PFN(input_dim=4,
          ppm_sizes=ppm_sizes,
          dense_sizes=dense_sizes,
          output_dim=1,
          output_act='linear',
          loss='mse',
          metrics=[accuracy, mean_pred])


##################################################################
#  _                    ______                _   _              #
# | |                   |  ___|              | | (_)             #
# | |     ___  ___ ___  | |_ _   _ _ __   ___| |_ _  ___  _ __   #
# | |    / _ \/ __/ __| |  _| | | | '_ \ / __| __| |/ _ \| '_ \  #
# | |___| (_) \__ \__ \ | | | |_| | | | | (__| |_| | (_) | | | | #
# \_____/\___/|___/___/ \_|  \__,_|_| |_|\___|\__|_|\___/|_| |_| #
##################################################################


def make_binned_loss(res, pred):
    x = binned_statistic(res, res, statistic='mean', bins=50)[0]
    y = binned_statistic(res, pred, statistic='std', bins=50)[0]
    fitfunc = lambda c , x: c[0]*np.sqrt(x)+c[1]*x+c[2]
    errfunc = lambda c , x, y: (y - fitfunc(c, x))
    out = leastsq(errfunc, [1., 0.1, 0.], args=(x, y), full_output=1)
    c = out[0]
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
    return binned_max_likelihood_loss

def make_loss(res, pred):
    x = binned_statistic(res, res, statistic='mean', bins=50)[0]
    y = binned_statistic(res, pred, statistic='std', bins=50)[0]
    fitfunc = lambda c , x: c[0]*np.sqrt(x)+c[1]*x+c[2]
    errfunc = lambda c , x, y: (y - fitfunc(c, x))
    out = leastsq(errfunc, [1., 0.1, 0.], args=(x, y), full_output=1)
    c = out[0]
    
    def likelihood_loss(y_true, y_pred):
        epsilon = tf.constant(0.0000001)
        mu = y_pred
        sigma = c[0]*tf.sqrt(y_true)+c[1]*y_true+c[2]
        first_part = tf.divide(tf.square(mu - y_true),
                               2.*tf.square(sigma)+epsilon)
        a = tf.divide(150.-mu, tf.sqrt(2.)*sigma+epsilon)
        b = tf.divide(30.-mu, tf.sqrt(2.)*sigma+epsilon)
        penalty = tf.erf(a) - tf.erf(b)
        loss = first_part + tf.log(penalty+epsilon) + tf.log(tf.sqrt(2.*np.pi)*sigma+epsilon)
        return tf.reduce_mean(loss)
    return likelihood_loss

#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

train_Gen = DataGenerator(batch_size=1024, train=True)
val_Gen = DataGenerator(batch_size=1024, train=False)

D = pfn.model

D.compile(loss='mse', optimizer='adagrad', metrics=[accuracy])

D.fit_generator(train_Gen, epochs=10*epochs, validation_data=val_Gen, validation_steps=len(val_Gen))


D.save_weights('pfn_weights.h5')


epochs = 1
train_Gen = DataGenerator(batch_size=1024, train=True, percent=0.1)
val_Gen = DataGenerator(batch_size=1024, train=False, percent=0.1)

pred = D.predict_generator(val_Gen)
res = val_Gen.dump_res()
pred = pred.reshape(len(pred),)

D.compile(loss=make_loss(res, pred), optimizer=RMSprop(lr=0.00001), metrics=[accuracy])

hist_update = D.fit_generator(train_Gen,
                              epochs=epochs,
                              validation_data=val_Gen,
                              validation_steps=len(val_Gen)).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

pred = D.predict_generator(val_Gen)
pred = pred.reshape(len(pred),)

D.save_weights('pfn_weights1.h5')


pred = D.predict_generator(val_Gen)
pred = pred.reshape(len(pred),)

D.compile(loss=make_loss(res, pred), optimizer=RMSprop(lr=0.00001), metrics=[accuracy])

hist_update = D.fit_generator(train_Gen,
                              epochs=epochs,
                              validation_data=val_Gen,
                              validation_steps=len(val_Gen)).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

pred = D.predict_generator(val_Gen)
pred = pred.reshape(len(pred),)

D.save_weights('pfn_weights2.h5')


pred = D.predict_generator(val_Gen)
pred = pred.reshape(len(pred),)

D.compile(loss=make_binned_loss(res, pred), optimizer=RMSprop(lr=0.00001), metrics=[accuracy])

hist_update = D.fit_generator(train_Gen,
                              epochs=epochs,
                              validation_data=val_Gen,
                              validation_steps=len(val_Gen)).history

history.update([('loss', history['loss'] + hist_update['loss']),
                ('val_loss', history['val_loss'] +
                 hist_update['val_loss'])])

D.save_weights('pfn_weights3.h5')

pickle.dump(history, open("pfn_history.p", "wb"))
