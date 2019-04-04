# this file is for outsourcing python functions, to keep our notebooks clean
import keras
import numpy as np
import random


def sliced_statistics(y_true, y_pred, number_of_slices):
    '''
    Sort the two arrays y_pred and y_true in y_pred after y_true. Cut
    the sorted arrays into n slices. Return the std and mean of the
    slices.
    y_true: numpy array with shape (x,) with x element of the natural numbers
    y_pred: numpy array with the same shape as y_pred
    n: len of the resulting arrays (int and lower than len(y_true))
    returns: arrays with values of value, mu, sigma

    '''
    # ensure(len(y_true.shape) == 2 and y_true.shape[1] == 1).is_(True)
    # ensure(y_true.shape == y_pred.shape).is_(True)
    # ensure(n).is_an(int)
    # ensure(n < len(y_true)).is_(True)

    # find the sorted order of y_true
    sliceable_length = len(y_true) - (len(y_true) % number_of_slices)
    sorted_order = y_true[:sliceable_length].argsort()
    length_of_slice = int(sliceable_length / number_of_slices)
    # calculate results
    y_true = y_true[sorted_order].reshape(number_of_slices, length_of_slice)
    y_pred = y_pred[sorted_order].reshape(number_of_slices, length_of_slice)
    value = np.mean(y_true, axis=1)
    mu = np.mean(y_pred, axis=1)
    sigma = np.std(y_pred, axis=1)
    return value, mu, sigma


class DataGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, z_set=None, batch_size=128, height=8,
                 width=8, channels=17, data_augment=True, adv=False, shape_learning=False):
        self.x, self.y, self.z = x_set, y_set, z_set
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels
        self.data_augment = data_augment
        self.adv = adv
        self.shape_learning = shape_learning

    def __len__(self):
        return int(np.floor(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = batch_x.reshape(self.batch_size, self.channels,
                                  self.height, self.width)
        # (batch_size, x, y, z) -> (batch_size, y, z, x) because xXS_HHHHH
        # represents the 'channels' if we interprete our data as
        # an image.

        batch_x = np.transpose(batch_x, (0, 2, 3, 1))

        # data_augmentation
        if self.data_augment:
            # randomly flip the array
            if random.choice([True, False]):
                batch_x = np.flip(batch_x, axis=1)
            # randomly turn the array around
            batch_x = np.rot90(batch_x, k=random.randint(0, 3), axes=(1, 2))

            # randomly shift the array entries
            zero_block = np.zeros(batch_x.shape)
            # first in heights
            max_shift = int(self.height/2)-1
            shift = random.randint(-max_shift, max_shift)
            batch_x = np.concatenate([zero_block, batch_x, zero_block], axis=1)
            batch_x = batch_x[:, self.height-shift:2*self.height-shift]
            # than in width
            max_shift = int(self.width/2)-1
            shift = random.randint(-max_shift, max_shift)
            batch_x = np.concatenate([zero_block, batch_x, zero_block], axis=2)
            batch_x = batch_x[:, :, self.width-shift:2*self.width-shift]

        batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], batch_x.shape[3], 1)

        if self.shape_learning:
            y_fit = (0.04*np.sum(batch_x, axis=1)-0.09).reshape(128, 1)
            return batch_x, [batch_y, y_fit]

        
        if self.z is None:
            return batch_x, batch_y

        else:
            batch_z = self.z[idx * self.batch_size:(idx + 1) * self.batch_size]
            if self.adv is True:
                return [batch_x, batch_y], batch_z
            else:
                return [batch_x, batch_y], [batch_y, batch_z]
