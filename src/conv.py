##############################################################
#    ___           _                                         #
#   / _ \__ _  ___| | ____ _  __ _  ___  ___                 #
#  / /_)/ _` |/ __| |/ / _` |/ _` |/ _ \/ __|                #
# / ___/ (_| | (__|   < (_| | (_| |  __/\__ \                #
# \/    \__,_|\___|_|\_\__,_|\__, |\___||___/                #
#                            |___/                           #
##############################################################

from keras.layers import Input, Dense, Flatten, Activation, Conv3D
from keras.models import Model
import h5py
import pickle
import numpy as np
from utils import DataGenerator
from utils import sliced_statistics
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

history = {'loss': [], 'val_loss': [],
           'weighted_loss': [], 'weighted_val_loss': [],
           'likeli_loss': [], 'likeli_val_loss': [],
           'da_loss': [], 'da_val_loss': [],
           'da_likeli_loss': [], 'da_likeli_val_loss': []}

y_pred = {}
y_true = {}
y_pred = {}
y = {}
mu = {}
sigma  = {}

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
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(128, activation="relu")(Dx)
Dx = Dense(10, activation="relu")(Dx)
Dx = Dense(1, activation="linear")(Dx)
D = Model([inputs], [Dx], name='D')
D.summary()

##################################################################
#  _                    ______                _   _              #
# | |                   |  ___|              | | (_)             #
# | |     ___  ___ ___  | |_ _   _ _ __   ___| |_ _  ___  _ __   #
# | |    / _ \/ __/ __| |  _| | | | '_ \ / __| __| |/ _ \| '_ \  #
# | |___| (_) \__ \__ \ | | | |_| | | | | (__| |_| | (_) | | | | #
# \_____/\___/|___/___/ \_|  \__,_|_| |_|\___|\__|_|\___/|_| |_| #
##################################################################
                                                              
def make_loss(c):
    def likelihood_loss(y_true, y_pred):
        epsilon = tf.constant(0.0000001)
        mu = y_pred
        sigma = c*tf.sqrt(y_true)
        first_part = tf.divide(tf.square(mu - y_true),
                               2.*tf.square(sigma)+epsilon)
        a = tf.divide(10.-mu, tf.sqrt(2.)*sigma+epsilon)
        b = tf.divide(0.-mu, tf.sqrt(2.)*sigma+epsilon)
        penalty = tf.erf(a) - tf.erf(b)
        loss = first_part + tf.log(penalty+epsilon) + tf.log(tf.sqrt(2.*np.pi)*sigma+epsilon)
        return tf.reduce_mean(loss)
    return likelihood_loss

def weighted_loss(y_true, y_pred):
    epsilon = tf.constant(0.0000001)
    mu = y_pred
    sigma = tf.sqrt(y_true)
    first_part = tf.divide(tf.square(mu - y_true),
                           2.*tf.square(sigma)+epsilon)
    return tf.reduce_mean(first_part)
 

#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

#############################################################
# train 
# the net with weighted loss
#############################################################

D.compile(loss=weighted_loss, optimizer='rmsprop')

initial_weights = D.get_weights()

epochs= 150

hist_update = D.fit_generator(
    DataGenerator(X_train, Y_train, batch_size=128, data_augment=True), epochs=epochs,
    validation_data=DataGenerator(X_test, Y_test, batch_size=128,
                                  data_augment=False), validation_steps=1).history

history.update([('weighted_loss', history['weighted_loss'] + hist_update['loss']),
                ('weighted_val_loss', history['weighted_val_loss'] +
                 hist_update['val_loss'])])
weights = D.get_weights()

n = 20
y_pred['weighted'] = D.predict_generator(DataGenerator(X_test, Y_test, batch_size=128, data_augment=False))
y_pred['weighted'] = y_pred['weighted'].reshape(len(y_pred['weighted']), )
y_true['weighted'] = np.array(Y_test)[:len(y_pred['weighted'])].reshape(len(y_pred['weighted']), )
y['weighted'], mu['weighted'], sigma['weighted'] = sliced_statistics(y_true['weighted'], y_pred['weighted'], n)


#############################################################
# train 
# train net with likelihood loss
#############################################################

# calculate factor for likelihood_loss
c = np.mean(sigma['weighted']/np.sqrt(mu['weighted']))
print('c is ' + str(c))
D.compile(loss=make_loss(c), optimizer='rmsprop')

epochs = 50

hist_update = D.fit_generator(
    DataGenerator(X_train, Y_train,
                  batch_size=128, data_augment=True), epochs=epochs,
    validation_data=DataGenerator(X_test, Y_test, batch_size=128,
                                  data_augment=False), validation_steps=1).history

history.update([('likeli_loss', history['likeli_loss'] + hist_update['loss']),
                ('likeli_val_loss', history['likeli_val_loss'] +
                 hist_update['val_loss'])])
likeli_weights = D.get_weights()

n = 20

y_pred['likeli'] = D.predict_generator(DataGenerator(X_test, Y_test, batch_size=128, data_augment=False))
y_pred['likeli'] = y_pred['likeli'].reshape(len(y_pred['likeli']), )
y_true['likeli'] = np.array(Y_test)[:len(y_pred['likeli'])].reshape(len(y_pred['likeli']), )
y['likeli'], mu['likeli'], sigma['likeli'] = sliced_statistics(y_true['likeli'], y_pred['likeli'], n)

#############################################################
# save results
results = {'y_pred': y_pred,
           'y_true': y_true,
           'y': y,
           'mu': mu,
           'sigma': sigma,
           'history': history}

pickle.dump(results, open("../results/conv_results.p", "wb"))
