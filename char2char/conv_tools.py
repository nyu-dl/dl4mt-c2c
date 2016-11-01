import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle
import numpy
import copy

import os
import warnings
import sys
import time

def conv_mask_pool(x_mask, pool_stride):
    # x_mask.shape = (maxlen_x_pad, n_samples)
    maxlen_x_pad, n_samples = x_mask.shape[0], x_mask.shape[1]
    maxlen_pooled = maxlen_x_pad / pool_stride

    x_m = numpy.zeros((maxlen_pooled, n_samples)).astype('float32')

    for idx in range(n_samples):
        x_sum = numpy.sum(x_mask[:,idx])
        x_num = numpy.ceil( x_sum  / float(pool_stride))
        x_num = int(x_num)
        x_m[:x_num, idx] = 1.0

    return x_m
