import os
import argparse
import sys
from collections import OrderedDict
from nmt import train
from wmt_path import wmts
from char_base import *

layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'fff': ('param_init_ffflayer', 'ffflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'two_layer_gru_decoder': ('param_init_two_layer_gru_decoder',
                                    'two_layer_gru_decoder'),
          }

def main(job_id, params):
    save_file_name = args.model_name
    source_dataset = args.data_path + wmts[args.translate]['train'][1][0]
    target_dataset = args.data_path + wmts[args.translate]['train'][0][1]
    valid_source_dataset = args.data_path + wmts[args.translate]['dev'][1][0]
    valid_target_dataset = args.data_path + wmts[args.translate]['dev'][0][1]
    source_dictionary = args.data_path + wmts[args.translate]['dic'][1][0]
    target_dictionary = args.data_path + wmts[args.translate]['dic'][0][1]

    print args.save_path, save_file_name
    print source_dataset
    print target_dataset
    print valid_source_dataset
    print valid_target_dataset
    print source_dictionary
    print target_dictionary
    print params, params.save_path, save_file_name

    validerr = train(
        max_epochs=args.max_epochs,
        patience=args.patience,

        dim_word_src=args.dim_word_src,
        dim_word=args.dim_word,

        save_path=args.save_path,
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

	use_dropout=args.use_dropout,
        source_word_level=args.source_word_level,
        target_word_level=args.target_word_level,
        save_every_saveFreq=1,
        use_bpe=1,
        gru=args.gru,

        quit_immediately=args.quit_immediately,
        init_params=init_params,
        build_model=build_model,
        build_sampler=build_sampler,
        gen_sample=gen_sample,
    )
    return validerr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, help="", default="bi-bpe2char")
    parser.add_argument('-translate', type=str, default="de_en", help="de_en / cs_en / fi_en / ru_en")

    parser.add_argument('-enc_dim', type=int, default=512, help="")
    parser.add_argument('-dec_dim', type=int, default=1024, help="")

    parser.add_argument('-dim_word', type=int, default=512, help="")
    parser.add_argument('-dim_word_src', type=int, default=512, help="")

    parser.add_argument('-batch_size', type=int, default=128, help="")
    parser.add_argument('-valid_batch_size', type=int, default=128, help="")

    parser.add_argument('-maxlen', type=int, default=50, help="")
    parser.add_argument('-maxlen_trg', type=int, default=500, help="")
    parser.add_argument('-maxlen_sample', type=int, default=500, help="")

    parser.add_argument('-re_load', action="store_true", default=False)
    parser.add_argument('-re_load_old_setting', action="store_true", default=False)
    parser.add_argument('-quit_immediately', action="store_true", default=False)

    parser.add_argument('-use_dropout', action="store_true", default=False)

    parser.add_argument('-max_epochs', type=int, default=1000000000000, help="")
    parser.add_argument('-patience', type=int, default=-1, help="")
    parser.add_argument('-learning_rate', type=float, default=0.0001, help="")

    parser.add_argument('-n_words_src', type=int, default=302, help="298 for FI")
    parser.add_argument('-n_words', type=int, default=302, help="292 for FI")

    parser.add_argument('-optimizer', type=str, default="adam", help="")
    parser.add_argument('-decay_c', type=int, default=0, help="")
    parser.add_argument('-clip_c', type=int, default=1, help="")

    parser.add_argument('-gru', type=str, default="gru", help="gru/lngru")

    parser.add_argument('-saveFreq', type=int, default=5000, help="")
    parser.add_argument('-sampleFreq', type=int, default=5000, help="")
    parser.add_argument('-dispFreq', type=int, default=1000, help="")
    parser.add_argument('-validFreq', type=int, default=5000, help="")
    parser.add_argument('-pbatchFreq', type=int, default=5000, help="")
    parser.add_argument('-sort_size', type=int, default=20, help="")

    parser.add_argument('-source_word_level', type=int, default=1, help="")
    parser.add_argument('-target_word_level', type=int, default=0, help="")

    args = parser.parse_args()

    n_words_dic = {'de_en': [24254, 302], 'cs_en': [21816, 302], 'fi_en':[20783, 292], 'ru_en':[22106, 302]}

    args.n_words_src = n_words_dic[args.translate][0]
    args.n_words= n_words_dic[args.translate][1]

    args.save_path = "/misc/kcgscratch1/ChoGroup/jasonlee/dl4mt-c2c/models/" # change accordingly
    args.data_path = "/misc/kcgscratch1/ChoGroup/jasonlee/temp_data/wmt15/" # change accordingly
    args.save_path = args.save_path + args.translate + "/"

    main(0, args)
