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

from conv_tools import *

from collections import OrderedDict
from mixer import *

# batch preparation for char2char models
def prepare_data(seqs_x, seqs_y, pool_stride, maxlen=None, maxlen_trg=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen_trg:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None

    # n_samples is not always equal to batch_size, can be smaller!
    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) # SOS, EOS symbols are already added in data_iterator.py, hence no extra trick here.
    maxlen_y = numpy.max(lengths_y) + 1 # account for EOS symbol at the end of the target sentence.

    maxlen_x_pad = int( numpy.ceil( maxlen_x / float(pool_stride) ) * pool_stride )
    # 1st round padding, such that the length is a multiple of pool_stride

    x = numpy.zeros((maxlen_x_pad + 2*pool_stride, n_samples)).astype('int64')
    # 2nd round padding at the beginning & the end for consistency, because theano's "half convolution" pads with zero-vectors by default. We want to ensure we don't pad with actual zero vectors, but rather with PAD embeddings. This is for consistency. For more information, consult http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv2d

    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x_pad, n_samples)).astype('float32')

    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[ pool_stride : pool_stride + lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx], idx] = 1.

        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    x_m = conv_mask_pool(x_mask, pool_stride)
    # x_m.shape = (maxlen_x_pad/pool_stride, n_samples)
    # x_m is used as masks at the GRU layer, note its length is reduced by pool_stride.

    return x, x_m, y, y_mask, n_samples
