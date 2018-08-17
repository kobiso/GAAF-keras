from __future__ import division

from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

class GAAF(Activation):
    """Gradient Acceleration in Activation Function
    As described in https://arxiv.org/abs/1806.09783.
    """
    
    def __init__(self, activation, **kwargs):
        super(GAAF, self).__init__(activation, **kwargs)
        self.__name__ = 'GAAF'

def gaaf_relu(x):
    """GAAF for ReLU
    Shape function is a shifted sigmoid with shift parameter.
    """

    frequency = 10000
    shift = 4 # shape function shifting
    mut = x*frequency    
    gx = (mut-tf.floor(mut)-0.5)/frequency    
    # gx = (K.abs(mut-K.round(mut))-0.5)/frequency
    sx = K.sigmoid(x+shift)
    gaaf = K.relu(x) + (gx*sx)    
    
    return gaaf

def gaaf_tanh(x):
    """GAAF for Tanh
    Shape function is a modified Gaussian function with a peak point at y=1.
    """
    
    frequency=10000
    mut = x*frequency
    gx = (mut-tf.floor(mut)-0.5)/frequency
    mid = -K.pow(x,2)
    sx = K.exp(mid/3)
    gaaf = K.tanh(x) + gx*sx
    
    return gaaf

get_custom_objects().update({'GAAF_relu': GAAF(gaaf_relu)})
get_custom_objects().update({'GAAF_tanh': GAAF(gaaf_tanh)})