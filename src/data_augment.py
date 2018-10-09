import tensorflow as tf
import random
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size=32, height=8, width=8,
                 channels=17, random_flip=True, random_rotate=True,
                 random_shift_height=True, random_shift_width=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_shift_height = random_shift_height
        self.random_shift_width = random_shift_width

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # transform data to geometrical array

        batch_x = batch_x.reshape(self.batch_size, self.channels,
                                  self.height, self.width)
        # (batch_size, x, y, z) -> (batch_size, y, z, x) because x
        # represents the 'channels' if we interprete our data as
        # an image.
        batch_x = np.transpose(batch_x, (0, 2, 3, 1))
        # data_augmentation

        # randomly flip the array
        if self.random_flip:
            if random.choice([True, False]):
                batch_x = np.flip(batch_x, axis=1)

        # randomly turn the array around
        if self.random_rotate:
            batch_x = np.rot90(batch_x, k=random.randint(0, 3), axes=(1, 2))

        # randomly shift the array entries
        zero_block = np.zeros(batch_x.shape)
        # first in heights
        if self.random_shift_height:
            max_shift = int(self.height/2)-1
            shift = random.randint(-max_shift, max_shift)
            batch_x = np.concatenate([zero_block, batch_x, zero_block], axis=1)
            batch_x = batch_x[:,
                              self.height-shift:2*self.height-shift]

        # than in width
        if self.random_shift_width:
            max_shift = int(self.width/2)-1
            shift = random.randint(-max_shift, max_shift)
            batch_x = np.concatenate([zero_block, batch_x, zero_block], axis=2)
            batch_x = batch_x[:,
                              :,
                              self.width-shift:2*self.width-shift]

        return batch_x, batch_y


def _random_flip(data, flip_index):
    """
    Randomly (50% chance) flip an data along axis `flip_index`.
    Args:
      data: 4-D Tensor of shape `[batch, height, width, channels]`
      flip_index: The dimension along which to flip the data.
                  Vertical: 0, Horizontal: 1
    Returns:
      A tensor of the same type and shape as `data`.

    """
    uniform_random = random_ops.random_uniform(
        [array_ops.shape(data)[0]], 0, 1.0)
    mirror_cond = math_ops.less(uniform_random, .5)
    return array_ops.where(
        mirror_cond,
        data,
        functional_ops.map_fn(
            lambda x: array_ops.reverse(x, [flip_index]),
            data,
            dtype=data.dtype))


def random_flip(data):
    """
    Randomly (50% chance) flip an data along both axes.
    Args:
      data: 4-D Tensor of shape `[batch, height, width, channels]`
    Returns:
      A tensor of the same type and shape as `data`.

    """
    data = _random_flip(data, 0)
    return _random_flip(data, 1)


def _random_shift(data):
        '''
        Randomly shift `data` for max. height/2 and width/2 in all directions.
        Shifted places will be replaced by zeroes.
        Examples would be (without channels):
        _____     _____
        |123|     |000|
        |456| =>  |230|
        |789|     |560|
        -----     -----
        _____     _____
        |123|     |045|
        |456| =>  |078|
        |789|     |000|
        -----     -----
        __________      __________
        |00000000|      |00000120|
        |00012000|  =>  |00000340|
        |00034000|      |00000000|
        |00000000|      |00000000|
        ----------      ----------
        Args:
          data: 3-D Tensor of shape `[height, width, channels]`
        Returns:
          A tensor with the same shape type and shape as `data`

        '''
        zero_block = tf.zeros(data.shape, data.dtype)
        data_shape = data.shape.as_list()

        # shift in height
        half = int(data_shape[0]/2)
        shift = random.randint(-half, half)
        s_begin = data_shape[0]-shift
        s_end = s_begin + data_shape[0]
        data = tf.concat([zero_block, data, zero_block], 0)
        data = data[s_begin:s_end]
        # shift in width
        half = int(data_shape[1]/2)
        shift = random.randint(-half, half)
        s_begin = data_shape[1]-shift
        s_end = s_begin + data_shape[1]
        data = tf.concat([zero_block, data, zero_block], 1)
        data = data[:, s_begin:s_end]

        return data


def random_shift(data):
    '''
    Randomly shift `data` for max. height/2 and width/2 in all directions.
    Shifted places will be replaced by zeroes.
    Examples would be (without channels and batch_size):
    __________      __________
    |00000000|      |00000120|
    |00012000|  =>  |00000340|
    |00034000|      |00000000|
    |00000000|      |00000000|
    ----------      ----------
    _____     _____
    |123|     |000|
    |456| =>  |230|
    |789|     |560|
    -----     -----
    _____     _____
    |123|     |045|
    |456| =>  |078|
    |789|     |000|
    -----     -----

    Args:
        data: 4-D Tensor of shape `[batch_size, height, width, channels]`
    Returns:
        A tensor with the same shape type and shape as `data`

    '''
    return functional_ops.map_fn(
        lambda x: _random_shift(x),
        data,
        dtype=data.dtype)


def data_augment(data):
    data = tf.manip.reshape(data, (None, 17, 8, 8))
    data = tf.transpose(data, perm=[0, 2, 3, 1])
    data = random_flip(data)
    return random_shift(data)
