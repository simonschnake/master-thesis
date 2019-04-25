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
from keras.optimizers import SGD, RMSprop
from keras import backend as K
import h5py
import numpy as np
import pickle
from scipy.stats import binned_statistic
from scipy.optimize import leastsq
import tensorflow as tf
from utils import sliced_statistics
from energyflow.archs import PFN
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.labelsize'] = 15

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
            data = h5py.File('../../data/jets/QCD_Pt-30to150Run2Spring18.h5', 'r')
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
    def dump_res(self):
        return self.y[:, 7][:len(self)*self.batch_size]


##################################################################
#  _                    ______                _   _              #
# | |                   |  ___|              | | (_)             #
# | |     ___  ___ ___  | |_ _   _ _ __   ___| |_ _  ___  _ __   #
# | |    / _ \/ __/ __| |  _| | | | '_ \ / __| __| |/ _ \| '_ \  #
# | |___| (_) \__ \__ \ | | | |_| | | | | (__| |_| | (_) | | | | #
# \_____/\___/|___/___/ \_|  \__,_|_| |_|\___|\__|_|\___/|_| |_| #
##################################################################


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

##############################################################
#                        _      _                            #
#        /\/\   ___   __| | ___| |                           #
#       /    \ / _ \ / _` |/ _ \ |                           #
#      / /\/\ \ (_) | (_| |  __/ |                           #
#      \/    \/\___/ \__,_|\___|_|                           #
##############################################################

# network architecture parameters
ppm_sizes = (100, 100, 128)
dense_sizes = (100, 100, 100)

inputs = Input(shape=(200, 4,))
Dx = Flatten()(inputs)
Dx = Dense(800, activation="relu")(Dx)
Dx = Dense(700, activation="relu")(Dx)
Dx = Dense(600, activation="relu")(Dx)
yx = Dense(500, activation="relu")(Dx)
Dx = Dense(400, activation="relu")(Dx)
Dx = Dense(300, activation="relu")(Dx)
Dx = Dense(200, activation="relu")(Dx)
Dx = Dense(1, activation="linear")(Dx)
D = Model([inputs], [Dx], name='D')

n = 20 
def produce_results(save_name):
    pred = D.predict_generator(val_Gen)
    pred = pred.reshape(len(pred),)
    y, mu, sigma = sliced_statistics(results['y_true'], pred, n)
    res = {'pred': pred,
           'y': y,
           'mu': mu,
           'sigma': sigma,
           'history' : history}
    results[save_name] = res


#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

epochs = 5
train_Gen = DataGenerator(batch_size=1024, train=True)
val_Gen = DataGenerator(batch_size=1024, train=False)
rmsprop = RMSprop(lr=0.0001)

results = {}
results['y_true'] = val_Gen.dump_res()

###############################################################################

D.compile(loss='mse', optimizer=rmsprop, metrics=[accuracy])

history = D.fit_generator(train_Gen,
                          epochs=epochs,
                          validation_data=val_Gen,
                          validation_steps=3).history

produce_results('first')

pred = D.predict_generator(train_Gen)
res = train_Gen.dump_res()
pred = pred.reshape(len(pred),)

###############################################################################

D.compile(loss=make_binned_loss(res, pred), optimizer=rmsprop, metrics=[accuracy])

history = D.fit_generator(train_Gen,
                          epochs=epochs,
                          validation_data=val_Gen,
                          validation_steps=len(val_Gen)).history

produce_results('second')

pred = D.predict_generator(train_Gen)
res = train_Gen.dump_res()
pred = pred.reshape(len(pred),)

################################################################################

D.compile(loss=make_binned_loss(res, pred), optimizer=rmsprop, metrics=[accuracy])

history = D.fit_generator(train_Gen,
                          epochs=epochs,
                          validation_data=val_Gen,
                          validation_steps=len(val_Gen)).history

produce_results('third')

pred = D.predict_generator(train_Gen)
res = train_Gen.dump_res()
pred = pred.reshape(len(pred),)

################################################################################

D.compile(loss=make_binned_loss(res, pred), optimizer=rmsprop, metrics=[accuracy])

history = D.fit_generator(train_Gen,
                          epochs=epochs,
                          validation_data=val_Gen,
                          validation_steps=len(val_Gen)).history

produce_results('fourth')

################################################################################

pickle.dump(results, open("../results/jetnet_binned_results.p", "wb"))
