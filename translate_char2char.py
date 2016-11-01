import argparse
import sys
import os
import time

reload(sys)
sys.setdefaultencoding('utf-8')

import numpy
import cPickle as pkl
from mixer import *

def translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, norm_pow, silent):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    use_noise = theano.shared(numpy.float32(0.))
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        use_noise.set_value(0.)
        # sample given an input sequence and obtain scores
        # NOTE : if seq length too small, do something about it
        sample, score = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, maxlen=500,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample]) ** norm_pow
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    while jobqueue:
        req = jobqueue.pop(0)

        idx, x = req[0], req[1]
        if not silent:
            print "sentence", idx, model_id
        seq = _translate(x)

        resultqueue.append((idx, seq))
    return

def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, encoder_chr_level=False,
         decoder_chr_level=False, utf8=False, decoder_bpe_to_tok=False,
          model_id=None, norm_pow=1, silent=False):

    from char_base import (build_sampler, gen_sample, init_params)

    # load model model_options
    # /misc/kcgscratch1/ChoGroup/jasonlee/dl4mt-cdec/models/one-multiscale-conv-two-hw-lngru-1234567-100-150-200-200-200-200-200-66-one.pkl
    pkl_file = model.split('.')[0] + '.pkl'
    with open(pkl_file, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    #word_idict[0] = 'ZERO'
    #word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    #word_idict_trg[0] = 'ZERO'
    #word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    jobqueue = []
    resultqueue = []

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                if utf8:
                    ww.append(word_idict_trg[w].encode('utf-8'))
                else:
                    ww.append(word_idict_trg[w])
            if decoder_chr_level:
                capsw.append(''.join(ww))
            else:
                capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                # idx : 0 ... len-1 
                pool_window = options['pool_stride']

                if encoder_chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()

                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                x = [2] + x + [3]

                # len : 77, pool_window 10 -> 3 
                # len : 80, pool_window 10 -> 0
                #rem = pool_window - ( len(x) % pool_window )
                #if rem < pool_window:
                #    x += [0]*rem

                while len(x) % pool_window != 0:
                    x += [0]

                x = [0]*pool_window + x + [0]*pool_window

                jobqueue.append((idx, x))

        return idx+1

    def _retrieve_jobs(n_samples, silent):
        trans = [None] * n_samples

        for idx in xrange(n_samples):
            resp = resultqueue.pop(0)
            trans[resp[0]] = resp[1]
            if numpy.mod(idx, 10) == 0:
                if not silent:
                    print 'Sample ', (idx+1), '/', n_samples, ' Done', model_id
        return trans

    print 'Translating ', source_file, '...'
    n_samples = _send_jobs(source_file)
    print "jobs sent"

    translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, norm_pow, silent)
    trans = _seqs2words(_retrieve_jobs(n_samples, silent))
    print "translations retrieved"

    with open(saveto, 'w') as f:
        if decoder_bpe_to_tok:
            print >>f, u'\n'.join(trans).encode('utf-8').replace('@@ ', '')
        else:
            print >>f, u'\n'.join(trans).encode('utf-8')

    print "Done", saveto

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=20) # beam width
    parser.add_argument('-norm_pow', type=float, default=1.0)
    parser.add_argument('-n', action="store_true", default=True) # normalise
    parser.add_argument('-bpe', action="store_true", default=False) # bpe source?
    parser.add_argument('-enc_c', action="store_true", default=True)
    parser.add_argument('-dec_c', action="store_true", default=True)
    parser.add_argument('-utf8', action="store_true", default=True)
    parser.add_argument('-model', type=str) # path_to_model
    parser.add_argument('-translate', type=str, ) # de_en, cs_en, fi_en, ru_en
    parser.add_argument('-saveto', type=str, ) # destination folder
    parser.add_argument('-which', type=str, help="dev / test1 / test2")
    parser.add_argument('-many', action="store_true", default=False) # bilingual or multilingual?
    parser.add_argument('-silent', action="store_true", default=False) # print progress or not

    args = parser.parse_args()

    which_wmt = None
    if args.many:
        which_wmt = "multi-wmt15"
    else:
        which_wmt = "wmt15"

    data_path = "/local/home/leeyu/dataset/%s/" % which_wmt # change appropriately

    if args.which not in "dev test1 test2".split():
        raise Exception('1')

    if args.translate not in ["de_en", "cs_en", "fi_en", "ru_en"]:
        raise Exception('1')

    if args.translate == "fi_en" and args.which == "test2":
        raise Exception('1')

    if args.many:
        from wmt_path_iso9 import *

        dictionary = wmts['many_en']['dic'][0][0]
        dictionary_target = wmts['many_en']['dic'][0][1]
        source = wmts[args.translate][args.which][0][0]

    else:
        from wmt_path import *

        aa = args.translate.split("_")
        lang = aa[0]
        en = aa[1]

        dictionary = "%s%s/train/all_%s-%s.%s.tok.304.pkl" % (lang, en, lang, en, lang)
        dictionary_target = "%s%s/train/all_%s-%s.%s.tok.300.pkl" % (lang, en, lang, en, en)
        source = wmts[args.translate][args.which][0][0]

    char_base = args.model.split("/")[-1]

    dictionary = data_path + dictionary
    dictionary_target = data_path + dictionary_target
    source = data_path + source
    print "src dict:", dictionary
    print "trg dict:", dictionary_target
    print "source:", source
    print "dest :", args.saveto

    print args

    time1 = time.time()
    main(args.model, dictionary, dictionary_target, source,
         args.saveto, k=args.k, normalize=args.n, encoder_chr_level=args.enc_c,
         decoder_chr_level=args.dec_c,
         utf8=args.utf8, decoder_bpe_to_tok=args.bpe,
         model_id=char_base,
         norm_pow=args.norm_pow,
         silent=args.silent,
        )
    time2 = time.time()
    duration = (time2-time1)/float(60)
    print("Translation took %.2f minutes" % duration)
