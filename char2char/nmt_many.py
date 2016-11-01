'''
Build a simple neural machine translation model using GRU units
'''
import theano
import datetime
import sys
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

reload(sys)
sys.setdefaultencoding('utf-8')

import cPickle
import numpy
import copy
from print_batch import pbatch_many

import os
import warnings
import sys
import time

from collections import OrderedDict
from mixer import *

from data_iterator import TextIterator
from many_data_iterator import MultiTextIterator

# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, pool_stride, verbose=True, verboseFreq=None):
    # NOTE : iterator is ALWAYS valid
    probs = []

    n_done = 0
    cnt = 0

    for x, y in iterator:
        n_done += len(x)
        cnt += 1

        x, x_mask, y, y_mask, n_x = prepare_data(x, y, pool_stride)

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            import ipdb
            ipdb.set_trace()

        if verbose:
            if numpy.mod(cnt, verboseFreq) == 0:
                print >>sys.stderr, '%d samples computed' % (cnt * n_done)

    return numpy.array(probs)

def train(
      highway=2,
      dim_word=100,
      dim_word_src=200,
      enc_dim=1000,
      dec_dim=1000,  # the number of LSTM units
      model_name="model_name",
      conv_width=4,
      conv_nkernels=256,
      pool_window=-1,
      pool_stride=-1,
      patience=-1,  # early stopping patience
      max_epochs=5000,
      finish_after=-1,  # finish after this many updates
      decay_c=0.,  # L2 regularization penalty
      alpha_c=0.,  # alignment regularization
      clip_c=-1.,  # gradient clipping threshold
      lrate=0.01,  # learning rate
      n_words_src=100000,  # source vocabulary size
      n_words=100000,  # target vocabulary size
      maxlen=1000,  # maximum length of the description
      maxlen_trg=1000,  # maximum length of the description
      maxlen_sample=1000,
      optimizer='rmsprop',
      batch_size=[1,2,3,4],
      valid_batch_size=16,
      sort_size=20,
      model_path=None,
      save_file_name='model',
      save_best_models=0,
      dispFreq=100,
      validFreq=100,
      saveFreq=1000,   # save the parameters after every saveFreq updates
      sampleFreq=-1,
      pbatchFreq=-1,
      verboseFreq=10000,
      datasets=[
          'data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
      valid_datasets=['../data/dev/newstest2011.en.tok',
                      '../data/dev/newstest2011.fr.tok'],
      dictionaries=[
          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
      source_word_level=0,
      target_word_level=0,
      dropout_gru=False,
      dropout_softmax=False,
      re_load=False,
      re_load_old_setting=False,
      uidx=None,
      eidx=None,
      cidx=None,
      layers=None,
      save_every_saveFreq=0,
      save_burn_in=20000,
      use_bpe=0,
      quit_immediately=False,
      init_params=None,
      build_model=None,
      build_sampler=None,
      gen_sample=None,
      prepare_data=None,
      **kwargs
    ):

    # Model options
    model_options = locals().copy()
    del model_options['init_params']
    del model_options['build_model']
    del model_options['build_sampler']
    del model_options['gen_sample']
    del model_options['prepare_data']

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = cPickle.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    print 'Building model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = '%s%s.npz' % (model_path, save_file_name)
    best_file_name = '%s%s.best.npz' % (model_path, save_file_name)
    opt_file_name = '%s%s%s.npz' % (model_path, save_file_name, '.grads')
    best_opt_file_name = '%s%s%s.best.npz' % (model_path, save_file_name, '.grads')
    model_name = '%s%s.pkl' % (model_path, save_file_name)
    params = init_params(model_options)

    cnt = 0
    cnt_emb = 0
    conv_params, hw_params = 0, 0
    for kk, vv in params.iteritems():
        if (kk == "Wemb"):
            print kk, vv.size
            cnt_emb += vv.size
        if "conv" in kk:
            print kk, vv.size
            conv_params += vv.size
        if "hw" in kk:
            print kk, vv.size
            hw_params += vv.size
        cnt += vv.size
    print "# Total params:", cnt
    print "# Emb params:", cnt_emb
    print "# Conv params:", conv_params
    print "# HW params:", hw_params
    print "# Input params:", cnt_emb + conv_params + hw_params

    if quit_immediately:
        sys.exit(1)

    cPickle.dump(model_options, open(model_name, 'wb'))
    history_errs = [[],[],[],[]]

    # reload options
    # reload : False
    if re_load and os.path.exists(file_name):
        print 'You are reloading your experiment.. do not panic dude..'
        if re_load_old_setting:
            with open(model_name, 'rb') as f:
                models_options = cPickle.load(f)
        params = load_params(file_name, params)
        # reload history
        model = numpy.load(file_name)
        history_errs = list(lst.tolist() for lst in model['history_errs'])
        if uidx is None:
            uidx = model['uidx']
        if eidx is None:
            eidx = model['eidx']
        if cidx is None:
            try:
                cidx = model['cidx']
            except:
                cidx = 0
    else:
        if uidx is None:
            uidx = 0
        if eidx is None:
            eidx = 0
        if cidx is None:
            cidx = 0

    print 'Loading data'
    train = MultiTextIterator(source=datasets[0],
                         target=datasets[1],
                         source_dict=dictionaries[0],
                         target_dict=dictionaries[1],
                         n_words_source=n_words_src,
                         n_words_target=n_words,
                         source_word_level=source_word_level,
                         target_word_level=target_word_level,
                         batch_size=batch_size,
                         sort_size=sort_size)

    valid = [TextIterator(source=valid_dataset[0],
                         target=valid_dataset[1],
                         source_dict=dictionaries[0],
                         target_dict=dictionaries[1],
                         n_words_source=n_words_src,
                         n_words_target=n_words,
                         source_word_level=source_word_level,
                         target_word_level=target_word_level,
                         batch_size=valid_batch_size,
                         sort_size=sort_size) for valid_dataset in valid_datasets]

    # create shared variables for parameters
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    # NOTE : this is where we build the model
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler...\n',
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)
    #print 'Done'

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    # NOTE : f_log_probs : [x, x_mask, y, y_mask], cost
    print 'Done'

    if re_load: 
        use_noise.set_value(0.)
        valid_scores = []
        for ii, vv in enumerate(valid):

            valid_err = pred_probs(f_log_probs,
                                   prepare_data,
                                   model_options,
                                   vv,
                                   pool_stride,
                                   verboseFreq=verboseFreq,
                                  ).mean()
            valid_err = valid_err.mean()

            if numpy.isnan(valid_err):
                import ipdb
                ipdb.set_trace()

            print 'Reload sanity check: Valid ', valid_err

    cost = cost.mean()

    # apply L2 regularization on weights
    # decay_c : 0
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    # alpha_c : 0
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    # NOTE : why is this not referenced somewhere later?
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    if clip_c > 0:
        grads, not_finite, clipped = gradient_clipping(grads, tparams, clip_c)
    else:
        not_finite = 0
        clipped = 0

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    if re_load and os.path.exists(file_name):
        if clip_c > 0:
            f_grad_shared, f_update, toptparams = eval(optimizer)(lr, tparams, grads, inps, cost=cost,
                                                                  not_finite=not_finite, clipped=clipped,
                                                                  file_name=opt_file_name)
        else:
            f_grad_shared, f_update, toptparams = eval(optimizer)(lr, tparams, grads, inps, cost=cost,
                                                                  file_name=opt_file_name)
    else:
        # re_load = False, clip_c = 1
        if clip_c > 0:
            f_grad_shared, f_update, toptparams = eval(optimizer)(lr, tparams, grads, inps, cost=cost,
                                                                  not_finite=not_finite, clipped=clipped)
        else:
            f_grad_shared, f_update, toptparams = eval(optimizer)(lr, tparams, grads, inps, cost=cost)

            # f_grad_shared = theano.function(inp, [cost, not_finite, clipped], updates=gsup, profile=profile)

            # f_update = theano.function([lr], [], updates=updates,
            #                   on_unused_input='ignore', profile=profile)
            # toptparams

    print 'Done'

    print 'Optimization'
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    # Training loop
    ud_start = time.time()
    estop = False

    if re_load:
        print "Checkpointed minibatch number: %d" % cidx
        for cc in xrange(cidx):
            if numpy.mod(cc, 1000)==0:
                print "Jumping [%d / %d] examples" % (cc, cidx)
            train.next()

    for epoch in xrange(max_epochs):
        time0 = time.time()
        n_samples = 0
        NaN_grad_cnt = 0
        NaN_cost_cnt = 0
        clipped_cnt = 0
        update_idx = 0
        if re_load:
            re_load = 0
        else:
            cidx = 0

        for x, y in train:
        # NOTE : x, y are [sen1, sen2, sen3 ...] where sen_i are of different length
            update_idx += 1
            cidx += 1
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask, n_x = prepare_data(x,
                                                     y,
                                                     pool_stride,
                                                     maxlen=maxlen,
                                                     maxlen_trg=maxlen_trg,
                                                    )

            if uidx == 1 or ( numpy.mod(uidx, pbatchFreq) == 0 and pbatchFreq != -1 ):
                pbatch_many(x, worddicts_r[0], n_x)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                uidx = max(uidx, 0)
                continue

            n_samples += n_x

            # compute cost, grads and copy grads to shared variables

            if clip_c > 0:
                cost, not_finite, clipped = f_grad_shared(x, x_mask, y, y_mask)
            else:
                cost = f_grad_shared(x, x_mask, y, y_mask)

            if clipped:
                clipped_cnt += 1

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                import ipdb
                ipdb.set_trace()
                NaN_cost_cnt += 1

            if not_finite:
                import ipdb
                ipdb.set_trace()
                NaN_grad_cnt += 1
                continue

            # do the update on parameters
            f_update(lrate)

            if numpy.isnan(cost) or numpy.isinf(cost):
                continue

            if float(NaN_grad_cnt) > max_epochs * 0.5 or float(NaN_cost_cnt) > max_epochs * 0.5:
                print 'Too many NaNs, abort training'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                ud = time.time() - ud_start
                wps = n_samples / float(time.time() - time0)
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'NaN_in_grad', NaN_grad_cnt,\
                      'NaN_in_cost', NaN_cost_cnt, 'Gradient_clipped', clipped_cnt, 'UD ', ud, "%.2f sentences/s" % wps
                ud_start = time.time()

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0 and sampleFreq != -1:
                gen_list = [0, batch_size[0], batch_size[0]+batch_size[1],  batch_size[0]+batch_size[1]+batch_size[2]]
                gen_list = [ii for ii in gen_list if ii < n_x]

                for jj in gen_list:
                    # jj = min(5, n_samples)
                    stochastic = True
                    use_noise.set_value(0.)

                    # x : maxlen X n_samples
                    sample, score = gen_sample(tparams, f_init, f_next,
                                               x[:, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               maxlen=maxlen_sample,
                                               stochastic=stochastic,
                                               argmax=False)
                    print
                    print 'Source ', jj, ': ',
                    if source_word_level:
                        for vv in x[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[0]:
                                if use_bpe:
                                    print (worddicts_r[0][vv]).replace('@@', ''),
                                else:
                                    print worddicts_r[0][vv],
                            else:
                                print 'UNK',
                        print
                    else:
                        source_ = []
                        for ii, vv in enumerate(x[:, jj]):
                            if vv == 0 or vv == 2 or vv == 3:
                                continue

                            if vv in worddicts_r[0]:
                                source_.append(worddicts_r[0][vv])
                            else:
                                source_.append('UNK')
                        print "".join(source_)
                    print 'Truth ', jj, ' : ',
                    if target_word_level:
                        for vv in y[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                if use_bpe:
                                    print (worddicts_r[1][vv]).replace('@@', ''),
                                else:
                                    print worddicts_r[1][vv],
                            else:
                                print 'UNK',
                        print
                    else:
                        truth_ = []
                        for vv in y[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                truth_.append(worddicts_r[1][vv])
                            else:
                                truth_.append('UNK')
                        print "".join(truth_)
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    if target_word_level:
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                if use_bpe:
                                    print (worddicts_r[1][vv]).replace('@@', ''),
                                else:
                                    print worddicts_r[1][vv],
                            else:
                                print 'UNK',
                        print
                    else:
                        sample_ = []
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                sample_.append(worddicts_r[1][vv])
                            else:
                                sample_.append('UNK')
                        print "".join(sample_)
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                valid_scores = []
                for ii, vv in enumerate(valid):
                    use_noise.set_value(0.)

                    valid_errs = pred_probs(f_log_probs,
                                            prepare_data,
                                            model_options,
                                            vv,
                                            pool_stride,
                                            verboseFreq=verboseFreq,
                                           )
                    valid_err = valid_errs.mean()
                    valid_scores.append(valid_err)
                    history_errs[ii].append(valid_err)

                    # patience == -1, never happens
                    if len(history_errs[ii]) > patience and valid_err >= \
                            numpy.array(history_errs[ii])[:-patience].min() and patience != -1:
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    if numpy.isnan(valid_err):
                        import ipdb
                        ipdb.set_trace()

                cnt = 0
                for ii in xrange(4):
                    if uidx == 0 or valid_scores[ii] <= numpy.array(history_errs[ii]).min():
                        cnt += 1 # cnt : the number of language pairs whose negative-log-likelihood increased this epoch.

                if cnt >= 2:
                    best_p = unzip(tparams)
                    best_optp = unzip(toptparams)
                    bad_counter = 0

                if saveFreq != validFreq and save_best_models:
                    numpy.savez(best_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                                cidx=cdix, **best_p)
                    numpy.savez(best_opt_file_name, **best_optp)

                print 'Valid : DE {}\t CS {}\t FI {}\t RU {}'.format(valid_scores[0], valid_scores[1], valid_scores[2], valid_scores[3])

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if not os.path.exists(model_path):
                    os.mkdir(model_path)

                params = unzip(tparams)
                optparams = unzip(toptparams)
                numpy.savez(file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                            cidx=cidx, **params)
                numpy.savez(opt_file_name, **optparams)

                if save_every_saveFreq and (uidx >= save_burn_in):
                    this_file_name = '%s%s.%d.npz' % (model_path, save_file_name, uidx)
                    this_opt_file_name = '%s%s%s.%d.npz' % (model_path, save_file_name, '.grads', uidx)
                    numpy.savez(this_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                                cidx=cidx, **params)
                    numpy.savez(this_opt_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                                cidx=cidx, **params)
                    if best_p is not None and saveFreq != validFreq:
                        this_best_file_name = '%s%s.%d.best.npz' % (model_path, save_file_name, uidx)
                        numpy.savez(this_best_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                                    cidx=cidx, **best_p)
                print 'Done...',
                print 'Saved to %s' % file_name

            # finish after this many updates
            if uidx >= finish_after and finish_after != -1:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples
        lang_nos = (4535523, 12122376, 1926115, 2326893)
        lang_done = [x * update_idx for x in batch_size]
        lang_rem = [x - y for x,y in zip(lang_nos, lang_done)]
        print "Remaining : DE({}), CS({}), FI({}), RU({})".format(lang_rem[0], lang_rem[1], lang_rem[2], lang_rem[3])
        eidx += 1

        if estop:
            break

    use_noise.set_value(0.)

    valid_scores = []
    for ii, vv in enumerate(valid):

        valid_err = pred_probs(f_log_probs,
                               prepare_data,
                               model_options,
                               vv,
                               pool_stride,
                              ).mean()
        valid_scores.append(valid_err)

    print 'Valid : DE {}\t CS {}\t FI {}\t RU {}'.format(valid_scores[0], valid_scores[1], valid_scores[2], valid_scores[3])

    params = unzip(tparams)
    optparams = unzip(toptparams)
    file_name = '%s%s.%d.npz' % (model_path, save_file_name, uidx)
    opt_file_name = '%s%s%s.%d.npz' % (model_path, save_file_name, '.grads', uidx)
    numpy.savez(file_name, history_errs=history_errs, uidx=uidx, eidx=eidx, cidx=cidx, **params)
    numpy.savez(opt_file_name, **optparams)
    if best_p is not None and saveFreq != validFreq:
        best_file_name = '%s%s.%d.best.npz' % (model_path, save_file_name, uidx)
        best_opt_file_name = '%s%s%s.%d.best.npz' % (model_path, save_file_name, '.grads',uidx)
        numpy.savez(best_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx, cidx=cidx, **best_p)
        numpy.savez(best_opt_file_name, **best_optp)

    return valid_err

if __name__ == '__main__':
    pass
