Fully Character-Level Neural Machine Translation
==================================

Theano implementation of the models described in the paper [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/abs/1610.03017 "Fully Character-Level Neural Machine Translation without Explicit Segmentation").

We present code for training and decoding four different models:

1. bi-bpe2char (from [Chung et al., 2016](https://arxiv.org/abs/1603.06147 "Chung et al., 2016"), code slightly modified.)

2. bi-char2char

3. multi-bpe2char

4. multi-char2char

DEPENDENCIES
------------------
### Python
* Theano
* Numpy
* NLTK

For preprocessing and evaluation, we used some scripts from [MOSES](https://github.com/moses-smt/mosesdecoder "MOSES").

This code is based on [Subword-NMT](http://arxiv.org/abs/1508.07909 "Subword-NMT") and [dl4mt-cdec](https://github.com/nyu-dl/dl4mt-cdec "dl4mt-cdec").

DOWNLOADING DATASETS
------------------


TRAINING A MODEL
------------------

Code for training bpe2char models resides in `dl4mt-c2c/bpe2char`. 

To use GPUs, do

```bash
$ export THEANO_FLAGS=device=gpu,floatX=float32
```

before starting training.

1. bi-bpe2char
```bash
$ python bpe2char/train_bi_bpe2char.py -translate <LANGUAGE_PAIR>
```

2. bi-char2char
```bash
$ python train_bi_char2char.py -translate <LANGUAGE_PAIR>
```

3. multi-bpe2char
```bash
$ python bpe2char/train_multi_bpe2char.py 
```

4. multi-char2char
```bash
$ python train_multi_char2char.py 
```

To resume training a model from a checkpoint, simply append `-re_load` and `-re_load_old_setting` above. Make sure the checkpoint resides in the correct directory.

DECODING
------------------

1. bpe2char models
```bash
$ cd dl4mt-c2c
$ python translate_char2char.py -translate <LANGUAGE_PAIR> -saveto <DESTINATION> -which <VALID/TEST_SET>
```

2. char2char models
```bash
$ cd dl4mt-c2c
$ python translate_bpe2char.py -translate <LANGUAGE_PAIR> -saveto <DESTINATION> -which <VALID/TEST_SET>
```

To decode multilingual models, append `-many`.

EVALUATION
------------------

```
perl preprocess/multi-bleu.perl reference.txt < model_output.txt
```

TREATMENT OF CYRILLIC
------------------

```bash
$ python iso.py <FILE_TO_BE_CONVERTED>
```

PRE-TRAINED MODELS
------------------

CITATION
------------------

```
@article{Lee16,
  author    = {Jason Lee and Kyunghyun Cho and Thomas Hofmann},
  title     = {Fully Character-Level Neural Machine Translation without Explicit Segmentation},
  journal   = {CoRR},
  volume    = {abs/1610.03017},
  year      = {2016},
  url       = {http://arxiv.org/abs/1610.03017},
}
```
