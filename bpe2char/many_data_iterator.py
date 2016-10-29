import nltk
import numpy
import os
import random

import cPickle
import gzip
import codecs

from tempfile import mkstemp

random.seed(1029381209)

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class MultiTextIterator:
    """Simple Bitext iterator."""
    def __init__(self,
                 source, source_dict,
                 target=None, target_dict=None,
                 source_word_level=0,
                 target_word_level=0,
                 batch_size=[128,1,2,3],
                 job_id=0,
                 sort_size=20,
                 n_words_source=302,
                 n_words_target=302,
                 shuffle_per_epoch=False):

        self.source_files = source
        self.target_files = target

        self.sources = [fopen(s, 'r') for s in source]
        with open(source_dict, 'rb') as f:
            self.source_dict = cPickle.load(f)
            # one source dictionary

        self.targets = [fopen(t, 'r') for t in target]
        with open(target_dict, 'rb') as f:
            self.target_dict = cPickle.load(f)
            # one target dictionary

        self.source_word_level = source_word_level
        self.target_word_level = target_word_level
        self.batch_sizes = batch_size
        # list

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        self.shuffle_per_epoch = shuffle_per_epoch

        self.source_buffers = [[],[],[],[]]
        self.target_buffers = [[],[],[],[]]
        self.k = [bs * sort_size for bs in batch_size]
        # at once, fetch 20 items
        # we're good for 20 updates

        self.end_of_data = False
        self.job_id = job_id

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle_per_epoch:
            raise Exception("hi")
            # close current files
            for s in self.sources:
                s.close()

            if self.targets is None:
                self.shuffle([self.source_file])
                self.source = fopen(self.source_file + '.reshuf_%d' % self.job_id, 'r')
            else:
                for t in self.targets:
                    t.close()

                # shuffle *original* source files,
                self.shuffle([self.source_file, self.target_file])
                # open newly 're-shuffled' file as input
                self.source = fopen(self.source_file + '.reshuf_%d' % self.job_id, 'r')
                self.target = fopen(self.target_file + '.reshuf_%d' % self.job_id, 'r')
        else:
            for idx in xrange(4):
                self.sources[idx].seek(0)
                self.targets[idx].seek(0)

    @staticmethod
    def shuffle(files):
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')
        fds = [open(ff) for ff in files]
        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print >>tf, "|||".join(lines)
        [ff.close() for ff in fds]
        tf.close()
        tf = open(tpath, 'r')
        lines = tf.readlines()
        random.shuffle(lines)
        fds = [open(ff+'.reshuf','w') for ff in files]
        for l in lines:
            s = l.strip().split('|||')
            for ii, fd in enumerate(fds):
                print >>fd, s[ii]
        [ff.close() for ff in fds]
        os.remove(tpath)
        return

    def next(self):
        # if end_of_data reaches, stop for loop
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        sources = [[],[],[],[]]
        targets = [[],[],[],[]]
        # NOTE : this is the data to be used for "this" round of updates

        # fill buffer, if it's empty
        for idx in xrange(4):
            assert len(self.source_buffers[idx]) == len(self.target_buffers[idx]), 'Buffer size mismatch!'

        for idx in xrange(4):
            # NOTE : in buffer: don't put the whole dataset in... only for 'k' many updates
            # after 'k' updates, self.source_buffers[idx] will be empty, in which case we will put new things in

            #if len(self.source_buffers[idx]) == 0:
            if len(self.source_buffers[idx]) < self.batch_sizes[idx]:
            # NOTE : change this to : if less than one out...
                for k_ in xrange(self.k[idx]):

                    ss = self.sources[idx].readline()
                    # NOTE: self.sources is where we keep the RAW data
                    if ss == "":
                        break
                    if self.source_word_level:
                        ss = ss.strip().split()
                    else:
                        ss = ss.strip()
                        ss = list(ss.decode('utf8'))
                    self.source_buffers[idx].append(ss)

                    tt = self.targets[idx].readline()
                    if tt == "":
                        break
                    if self.target_word_level:
                        tt = tt.strip().split()
                    else:
                        tt = tt.strip()
                        tt = list(tt.decode('utf8'))
                    self.target_buffers[idx].append(tt)

                tlen = numpy.array([len(t) for t in self.target_buffers[idx]])
                tidx = tlen.argsort()
                _sbuf = [self.source_buffers[idx][i] for i in tidx]
                _tbuf = [self.target_buffers[idx][i] for i in tidx]
                self.target_buffers[idx] = _tbuf
                self.source_buffers[idx] = _sbuf

        stop = False
        for idx in xrange(4):
            if len(self.source_buffers[idx]) < self.batch_sizes[idx]:
                stop = True

        if stop:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            for idx in xrange(4):
                while True:
                # read from source file and map to word index
                    try:
                        ss_ = self.source_buffers[idx].pop()
                    except IndexError:
                        # NOTE : just because source_buffers is empty, doesn't mean file scanned
                        # we do add partial batches. We proceed until len(source_buffers) = 0
                        break

                    ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss_]
                    if self.n_words_source > 0:
                        ss = [w if w < self.n_words_source else 1 for w in ss]
                    sources[idx].append(ss)

                    tt_ = self.target_buffers[idx].pop()
                    tt = [self.target_dict[w] if w in self.target_dict else 1 for w in tt_]
                    if self.n_words_target > 0:
                        tt = [w if w < self.n_words_target else 1 for w in tt]
                    targets[idx].append(tt)

                    if len(sources[idx]) >= self.batch_sizes[idx]:
                        break

        except IOError:
            self.end_of_data = True

        source = sources[0] + sources[1] + sources[2] + sources[3]
        target = targets[0] + targets[1] + targets[2] + targets[3]

        # NOTE : just add anything, if still nothing, reset
        min_batch_size = numpy.sum(self.batch_sizes)
        # NOTE : this CANT BE ZERO!!!! bc buffer not multiple of things
        if len(source) < min_batch_size or len(target) < min_batch_size:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target
