import tensorflow as tf
import random
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


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
