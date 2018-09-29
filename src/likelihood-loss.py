##############################################################
#    ___           _                                         #
#   / _ \__ _  ___| | ____ _  __ _  ___  ___                 #
#  / /_)/ _` |/ __| |/ / _` |/ _` |/ _ \/ __|                #
# / ___/ (_| | (__|   < (_| | (_| |  __/\__ \                #
# \/    \__,_|\___|_|\_\__,_|\__, |\___||___/                #
#                            |___/                           #
##############################################################
import numpy as np
from keras.optimizers import Adadelta
import h5py
import pickle
import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model

##############################################################
#  _                 _ _                  ___      _         #
# | | ___   __ _  __| (_)_ __   __ _     /   \__ _| |_ __ _  #
# | |/ _ \ / _` |/ _` | | '_ \ / _` |   / /\ / _` | __/ _` | #
# | | (_) | (_| | (_| | | | | | (_| |  / /_// (_| | || (_| | #
# |_|\___/ \__,_|\__,_|_|_| |_|\__, | /___,' \__,_|\__\__,_| #
#                              |___/                         #
##############################################################

data = h5py.File('../../data/electron.h5', 'r')
X_train = data['train']['X']
Y_train = data['train']['Y']

history = {'loss': [], 'val_loss': []}

##############################################################
#                        _      _                            #
#        /\/\   ___   __| | ___| |                           #
#       /    \ / _ \ / _` |/ _ \ |                           #
#      / /\/\ \ (_) | (_| |  __/ |                           #
#      \/    \/\___/ \__,_|\___|_|                           #
##############################################################


def likelihoodModel():
    # Define the input placeholder as a tensor with shape
    # input_shape. Think of this as your input image
    X_input = Input(shape=(1088,))
    X = Dense(128, activation='relu', name='fc')(X_input)
    X = Dense(128, activation='relu', name='fc1')(X)
    X = Dense(128, activation='relu', name='fc2')(X)
    X = Dense(10, activation='relu', name='fc3')(X)
    X = Dense(1, activation='linear', name='exit')(X)
    return Model(inputs=X_input, outputs=X, name='likelihoodModel')


def likelihood_loss(y_true, y_pred):
    epsilon = tf.constant(np.float64(0.0000001))
    lower_border = tf.constant(np.float64(0.))
    upper_border = tf.constant(np.float64(10.))
    two = tf.constant(np.float64(2.))
    pi = tf.constant(np.float64(np.pi))
    mu = y_pred
    sigma = tf.sqrt(tf.abs(y_true))
    elements = tf.divide(tf.exp(tf.divide(- tf.square(mu - y_true),
                                          two*tf.square(sigma))),
                         tf.sqrt(two*pi)*sigma)
    a = tf.divide(mu-lower_border, tf.sqrt(two)*sigma+epsilon)
    b = tf.divide(mu-upper_border, tf.sqrt(two)*sigma+epsilon)
    norms = tf.abs(tf.erf(a) - tf.erf(b))
    return -tf.reduce_mean(tf.log(tf.divide(elements,
                                            norms + epsilon)
                                  + epsilon))


# try:
#     model = load_model('model.h5')
# except IOError:
#     print('model not found')
#     exit()

# try:
#     history = pickle.load(open('history.p', 'rb'))
# except IOError:
#     print('no history file')
#     history = {'loss': [], 'val_loss': []}

#############################################################
#  _____           _       _                   __     _     #
# /__   \_ __ __ _(_)_ __ (_)_ __   __ _    /\ \ \___| |_   #
#   / /\/ '__/ _` | | '_ \| | '_ \ / _` |  /  \/ / _ \ __|  #
#  / /  | | | (_| | | | | | | | | | (_| | / /\  /  __/ |_   #
#  \/   |_|  \__,_|_|_| |_|_|_| |_|\__, | \_\ \/ \___|\__|  #
#                                  |___/                    #
#############################################################

model = likelihoodModel()

model.compile(optimizer='rmsprop',
              loss='mse')
epochs = 3
batch_size = 128
model.fit(X_train,
          Y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.2)


opt = Adadelta(lr=0.1)
model.compile(optimizer=opt, loss=likelihood_loss)

epochs = 5
batch_size = 1024
hist_update = model.fit(X_train,
                        Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2).history
history.update([('loss',
                 history['loss'] + hist_update['loss']),
                ('val_loss',
                 history['val_loss'] + hist_update['val_loss'])])


##############################################################
#  __             _                  ___      _              #
# / _\ __ ___   _(_)_ __   __ _     /   \__ _| |_ __ _       #
# \ \ / _` \ \ / / | '_ \ / _` |   / /\ / _` | __/ _` |      #
# _\ \ (_| |\ V /| | | | | (_| |  / /_// (_| | || (_| |      #
# \__/\__,_| \_/ |_|_| |_|\__, | /___,' \__,_|\__\__,_|      #
#                         |___/                              #
##############################################################

pickle.dump(history, open("likelihood_history.p", "wb"))

model.save_weights('likelihood_weights.h5')
