import os
import sys
import argparse
import string
import math
import numpy
from char_base_multi_b2c import *
from nmt_many import train

save_path = "/local/home/leeyu/scratch/dl4mt-c2c/models/"
data_path = "/local/home/leeyu/dataset/multi-wmt15/"
dic_path = "/local/home/leeyu/dataset/multi-wmt15/dic/"

from collections import OrderedDict

def main(job_id, params, args):
    print args
    save_file_name = args.model_name
    source_dataset = [data_path + path + tr for path, tr in zip(params['train_data_path'], params['source_dataset'])]
    target_dataset = [data_path + path + tr for path, tr in zip(params['train_data_path'], params['target_dataset'])]

    valid_source_dataset = [data_path + path + tr for path, tr in zip(params['dev_data_path'], params['valid_source_dataset'])]
    valid_target_dataset = [data_path + path + tr for path, tr in zip(params['dev_data_path'], params['valid_target_dataset'])]

    source_dictionary = dic_path + args.source_dictionary
    target_dictionary = dic_path + args.target_dictionary

    global save_path
    save_path = save_path + "many_en" + "/"

    print params, save_path, save_file_name
    validerr = train(
        max_epochs=int(params['max_epochs']),
        patience=int(params['patience']),

        dim_word=args.dim_word,
        dim_word_src=args.dim_word_src,

        save_path=save_path,
        save_file_name=save_file_name,
        re_load=args.re_load,
        re_load_old_setting=args.re_load_old_setting,

        enc_dim=args.enc_dim,
        dec_dim=args.dec_dim,

        n_words=args.n_words,
        n_words_src=args.n_words_src,
        decay_c=float(params['decay_c']),
        lrate=float(params['learning_rate']),
        optimizer=params['optimizer'],
        maxlen=args.maxlen,
        maxlen_trg=args.maxlen_trg,
        maxlen_sample=args.maxlen_sample,
        batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        sort_size=args.sort_size,
        validFreq=args.validFreq,
        dispFreq=args.dispFreq,
        saveFreq=args.saveFreq,
        sampleFreq=args.sampleFreq,
        pbatchFreq=args.pbatchFreq,
        clip_c=int(params['clip_c']),

        datasets=[source_dataset, target_dataset],
        valid_datasets=[[s,t] for s,t in zip(valid_source_dataset, valid_target_dataset)],
        dictionaries=[source_dictionary, target_dictionary],

        use_dropout=int(params['use_dropout']),
        source_word_level=int(params['source_word_level']),
        target_word_level=int(params['target_word_level']),
        save_every_saveFreq=1,
        use_bpe=0,
        init_params=init_params,
        build_model=build_model,
        build_sampler=build_sampler,
        gen_sample=gen_sample,
    )
    return validerr

if __name__ == '__main__':

    import sys, time

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, help="", default="multi-bpe2char")

    parser.add_argument('-enc_dim', type=int, default=512, help="")
    parser.add_argument('-dec_dim', type=int, default=1024, help="")

    parser.add_argument('-dim_word', type=int, default=512, help="")
    parser.add_argument('-dim_word_src', type=int, default=512, help="")

    parser.add_argument('-n_words', type=int, default=402, help="")
    parser.add_argument('-n_words_src', type=int, default=54541, help="")

    parser.add_argument('-source_dictionary', type=str, default="bpe-source-for-dic.word.pkl", help="")
    parser.add_argument('-target_dictionary', type=str, default="target.402.pkl", help="")

    parser.add_argument('-saveFreq', type=int, default=5000, help="")
    parser.add_argument('-sampleFreq', type=int, default=5000, help="")
    parser.add_argument('-dispFreq', type=int, default=1000, help="")
    parser.add_argument('-validFreq', type=int, default=5000, help="")
    parser.add_argument('-pbatchFreq', type=int, default=-1, help="")
    parser.add_argument('-sort_size', type=int, default=20, help="")

    parser.add_argument('-maxlen', type=int, default=50, help="")
    parser.add_argument('-maxlen_trg', type=int, default=500, help="")
    parser.add_argument('-maxlen_sample', type=int, default=500, help="")

    parser.add_argument('-train_batch_size', type=str, default="4535523/12122376/1926115/2326893", help="")
    parser.add_argument('-valid_batch_size', type=int, default=60, help="")
    parser.add_argument('-batch_size', type=int, default=60, help="")

    parser.add_argument('-re_load', action="store_true", default=False)
    parser.add_argument('-re_load_old_setting', action="store_true", default=False)

    args = parser.parse_args()

    args.model_name = args.model_name + "-" + str(args.enc_dim)

    args.train_batch_size = [ int(x) for x in args.train_batch_size.split("/") ]

    train_batch_sum = numpy.sum(args.train_batch_size)

    args.train_batch_size = [ int(numpy.ceil(args.batch_size * x / float(train_batch_sum))) for x in args.train_batch_size ]
    args.train_batch_size = [ 14, 37, 6, 7 ]

    config_file_name = '/local/home/leeyu/scratch/dl4mt-c2c/multi-bpe2char-code/wmt15_manyen_bpe2char_adam.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = OrderedDict()

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')

        if len(param_list) < 2:
            continue
        elif len(param_list) == 2:
            param_name = param_list[0]
            param_value = param_list[1]
            params[param_name] = param_value
        else:
            param_name = param_list[0]
            param_value = param_list[1:]
            params[param_name] = param_value

    main(0, params, args)
