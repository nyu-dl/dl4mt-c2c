import os
import sys
import argparse
import string
from collections import OrderedDict
from wmt_path import *
from char_base import *
from nmt import train
from conv_tools import *
from prepare_data import *

def main(job_id, args):
    save_file_name = args.model_name
    source_dataset = args.data_path + wmts[args.translate]['train'][0][0]
    target_dataset = args.data_path + wmts[args.translate]['train'][0][1]
    valid_source_dataset = args.data_path + wmts[args.translate]['dev'][0][0]
    valid_target_dataset = args.data_path + wmts[args.translate]['dev'][0][1]
    source_dictionary = args.data_path + wmts[args.translate]['dic'][0][0]
    target_dictionary = args.data_path + wmts[args.translate]['dic'][0][1]

    print args.model_path, save_file_name
    print source_dataset
    print target_dataset
    print valid_source_dataset
    print valid_target_dataset
    print source_dictionary
    print target_dictionary
    validerr = train(
        highway=args.highway,

        max_epochs=args.max_epochs,
        patience=args.patience,

        dim_word_src=args.dim_word_src,
        dim_word=args.dim_word,

        conv_width=args.conv_width,
        conv_nkernels=args.conv_nkernels,

        pool_window=args.pool_window,
        pool_stride=args.pool_stride,

        model_path=args.model_path,
        save_file_name=save_file_name,
        re_load=args.re_load,
        re_load_old_setting=args.re_load_old_setting,

        enc_dim=args.enc_dim,
        dec_dim=args.dec_dim,

        n_words_src=args.n_words_src,
        n_words=args.n_words,
        decay_c=args.decay_c,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        maxlen=args.maxlen,
        maxlen_trg=args.maxlen_trg,
        maxlen_sample=args.maxlen_sample,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        sort_size=args.sort_size,
        validFreq=args.validFreq,
        dispFreq=args.dispFreq,
        saveFreq=args.saveFreq,
        sampleFreq=args.sampleFreq,
        pbatchFreq=args.pbatchFreq,
        clip_c=args.clip_c,

        datasets=[source_dataset, target_dataset],
        valid_datasets=[valid_source_dataset, valid_target_dataset],
        dictionaries=[source_dictionary, target_dictionary],

        dropout_gru=args.dropout_gru,
        dropout_softmax=args.dropout_softmax,
        source_word_level=args.source_word_level,
        target_word_level=args.target_word_level,
        save_every_saveFreq=1,
        use_bpe=0,
        quit_immediately=args.quit_immediately,
        init_params=init_params,
        build_model=build_model,
        build_sampler=build_sampler,
        gen_sample=gen_sample,
        prepare_data=prepare_data,
    )
    return validerr

if __name__ == '__main__':

    import sys, time

    parser = argparse.ArgumentParser()
    parser.add_argument('-translate', type=str, default="de_en", help="de_en / cs_en / fi_en / ru_en")
    parser.add_argument('-highway', type=int, default=4)

    parser.add_argument('-conv_width', type=str, default="1-2-3-4-5-6-7-8")
    parser.add_argument('-conv_nkernels', type=str, default="200-200-250-250-300-300-300-300")

    parser.add_argument('-pool_window', type=int, default=5)
    parser.add_argument('-pool_stride', type=int, default=5)

    parser.add_argument('-enc_dim', type=int, default=512)
    parser.add_argument('-dec_dim', type=int, default=1024)

    parser.add_argument('-dim_word', type=int, default=512)
    parser.add_argument('-dim_word_src', type=int, default=128)

    parser.add_argument('-batch_size', type=int, default=64, help="")
    parser.add_argument('-valid_batch_size', type=int, default=64, help="")

    parser.add_argument('-dropout_gru', type=int, default=0, help="")
    parser.add_argument('-dropout_softmax', type=int, default=0, help="")

    parser.add_argument('-maxlen', type=int, default=450, help="")
    parser.add_argument('-maxlen_trg', type=int, default=500, help="")
    parser.add_argument('-maxlen_sample', type=int, default=500, help="")

    parser.add_argument('-re_load', action="store_true", default=False)
    parser.add_argument('-re_load_old_setting', action="store_true", default=False)
    parser.add_argument('-quit_immediately', action="store_true", default=False, help="if true, will not proceed training, only print the size of the model.")

    parser.add_argument('-max_epochs', type=int, default=1000000000000, help="")
    parser.add_argument('-patience', type=int, default=-1, help="")
    parser.add_argument('-learning_rate', type=float, default=0.0001, help="")

    parser.add_argument('-n_words_src', type=int, default=304, help="298 for FI-EN")
    parser.add_argument('-n_words', type=int, default=302, help="292 for FI-EN")

    parser.add_argument('-optimizer', type=str, default="adam", help="")
    parser.add_argument('-decay_c', type=int, default=0, help="")
    parser.add_argument('-clip_c', type=int, default=1, help="")

    parser.add_argument('-saveFreq', type=int, default=5000, help="")
    parser.add_argument('-sampleFreq', type=int, default=5000, help="")
    parser.add_argument('-dispFreq', type=int, default=1000, help="")
    parser.add_argument('-validFreq', type=int, default=5000, help="")
    parser.add_argument('-pbatchFreq', type=int, default=5000, help="")
    parser.add_argument('-sort_size', type=int, default=20, help="")

    parser.add_argument('-source_word_level', type=int, default=0, help="")
    parser.add_argument('-target_word_level', type=int, default=0, help="")

    args = parser.parse_args()

    if args.translate == "fi_en":
        args.n_words_src = 304
        args.n_words = 302

    if args.translate not in "de_en cs_en fi_en ru_en".split():
        raise Exception('1')

    args.model_name = "bi-char2char"

    args.conv_width = [ int(x) for x in args.conv_width.split("-") ]
    args.conv_nkernels = [ int(x) for x in args.conv_nkernels.split("-") ]

    args.model_path = "/misc/kcgscratch1/ChoGroup/jasonlee/dl4mt-c2c/models/" # change accordingly
    args.data_path = "/misc/kcgscratch1/ChoGroup/jasonlee/temp_data/wmt15/" # change accordingly
    args.model_path = args.model_path + args.translate + "/"

    print "Model path:", args.model_path

    print args
    main(0, args)
