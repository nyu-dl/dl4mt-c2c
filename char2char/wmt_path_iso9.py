#!/usr/bin/python
# Paths to training / valid / test corpus & dictionaries
# For the multilingual models (russian characters converted to latin using iso9)

deen={
    "dic": [
                ["deen/train/all_de-en.de.tok.shuf.iso9.304.pkl",
                 "dic/target.402.pkl"],

                ["deen/train/all_de-en.de.tok.shuf.iso9.bpe.24111.word.pkl"],
            ],

    "train": [
                ["deen/train/all_de-en.de.tok.shuf.iso9",
                 "deen/train/all_de-en.en.tok.shuf.iso9",],

                ["deen/train/all_de-en.de.tok.shuf.iso9.bpe.50000",
                 "deen/train/all_de-en.de.tok.shuf.iso9.bpe.20000"],
            ],

    "dev": [
                ["deen/dev/newstest2013.de.tok.iso9",
                 "deen/dev/newstest2013.en.tok.iso9",],

                ["deen/dev/newstest2013.de.tok.iso9.bpe.50000",
                 "deen/dev/newstest2013.de.tok.iso9.bpe.20000"],
            ],

    "test1" :[
                ["deen/test/newstest2014-deen-ref.de.tok.iso9",
                 "deen/test/newstest2014-deen-src.en.tok.iso9",],

                ["deen/test/newstest2014-deen-ref.de.tok.iso9.bpe.50000",
                 "deen/test/newstest2014-deen-ref.de.tok.iso9.bpe.20000"],
            ],

    "test2":[
                ["deen/test/newstest2015-deen-src.de.tok.iso9",
                 "deen/test/newstest2015-deen-src.en.tok.iso9",],

                ["deen/test/newstest2015-deen-src.de.tok.iso9.bpe.50000",
                 "deen/test/newstest2015-deen-src.de.tok.iso9.bpe.20000"],
            ],
}

csen={
    "dic": [
                ["csen/train/all_cs-en.cs.tok.iso9.304.pkl",
                 "dic/target.402.pkl"],

                ["csen/train/all_cs-en.cs.tok.iso9.bpe.21697.word.pkl"],
            ],

    "train":[
                ["csen/train/all_cs-en.cs.tok.iso9",
                 "csen/train/all_cs-en.en.tok.iso9",],

                ["csen/train/all_cs-en.cs.tok.iso9.bpe.50000",
                 "csen/train/all_cs-en.cs.tok.iso9.bpe.20000"],
            ],

    "dev": [
                ["csen/dev/newstest2013-ref.cs.tok.iso9",
                 "csen/dev/newstest2013-src.en.tok.iso9",],

                ["csen/dev/newstest2013-ref.cs.tok.iso9.bpe.50000",
                 "csen/dev/newstest2013-ref.cs.tok.iso9.bpe.20000"],
            ],

    "test1":[
                ["csen/test/newstest2014-csen-ref.cs.tok.iso9",
                 "csen/test/newstest2014-csen-src.en.tok.iso9",],

                ["csen/test/newstest2014-csen-ref.cs.tok.iso9.bpe.50000",
                 "csen/test/newstest2014-csen-ref.cs.tok.iso9.bpe.20000"],
        ],

    "test2":[
                ["csen/test/newstest2015-csen-ref.cs.tok.iso9",
                 "csen/test/newstest2015-csen-src.en.tok.iso9",],

                ["csen/test/newstest2015-csen-ref.cs.tok.iso9.bpe.50000",
                 "csen/test/newstest2015-csen-ref.cs.tok.iso9.bpe.20000"],
        ]
}

fien={
    "dic": [
                ["fien/train/all_fi-en.fi.tok.shuf.iso9.269.pkl",
                 "dic/target.402.pkl"],

                ["fien/train/all_fi-en.fi.tok.shuf.iso9.bpe.20747.word.pkl"],
            ],

    "train":[
                ["fien/train/all_fi-en.fi.tok.shuf.iso9",
                 "fien/train/all_fi-en.en.tok.shuf.iso9",],

                ["fien/train/all_fi-en.fi.tok.shuf.iso9.bpe.50000",
                 "fien/train/all_fi-en.fi.tok.shuf.iso9.bpe.20000"],
        ],

    "dev":[
                ["fien/dev/newsdev2015-enfi-ref.fi.tok.iso9",
                 "fien/dev/newsdev2015-enfi-src.en.tok.iso9",],

                ["fien/dev/newsdev2015-enfi-ref.fi.tok.iso9.bpe.50000",
                 "fien/dev/newsdev2015-enfi-ref.fi.tok.iso9.bpe.20000"],
        ],

    "test1":[
                ["fien/test/newstest2015-fien-ref.fi.tok.iso9",
                 "fien/test/newstest2015-fien-src.en.tok.iso9",],

                ["fien/test/newstest2015-fien-ref.fi.tok.iso9.bpe.50000",
                 "fien/test/newstest2015-fien-ref.fi.tok.iso9.bpe.20000"],
        ],
}

ruen={
    "dic": [
                ["ruen/train/all_ru-en.ru.tok.iso9.304.pkl",
                 "dic/target.402.pkl"],

                ["ruen/train/all_ru-en.ru.tok.iso9.bpe.21995.word.pkl"],
            ],

    "train":[
                ["ruen/train/all_ru-en.ru.tok.iso9",
                 "ruen/train/all_ru-en.en.tok.iso9",],

                ["ruen/train/all_ru-en.ru.tok.iso9.bpe.50000",
                 "ruen/train/all_ru-en.ru.tok.iso9.bpe.20000"],
        ],

    "dev":[
                ["ruen/dev/newstest2013-ref.ru.tok.iso9",
                 "ruen/dev/newstest2013-src.en.tok.iso9",],

                ["ruen/dev/newstest2013-ref.ru.tok.iso9.bpe.50000",
                 "ruen/dev/newstest2013-ref.ru.tok.iso9.bpe.20000"],
        ],

    "test1":[
                ["ruen/test/newstest2014-ruen-ref.ru.tok.iso9",
                 "ruen/test/newstest2014-ruen-src.en.tok.iso9",],

                ["ruen/test/newstest2014-ruen-ref.ru.tok.iso9.bpe.50000",
                 "ruen/test/newstest2014-ruen-ref.ru.tok.iso9.bpe.20000"],
        ],

    "test2":[
                ["ruen/test/newstest2015-ruen-ref.ru.tok.iso9",
                 "ruen/test/newstest2015-ruen-src.en.tok.iso9",],

                ["ruen/test/newstest2015-ruen-ref.ru.tok.iso9.bpe.50000",
                 "ruen/test/newstest2015-ruen-ref.ru.tok.iso9.bpe.20000"],
        ]
}

manyen = {
    "dic":[
            ["dic/source.404.pkl",
             "dic/target.402.pkl"],

            ["dic/bpe-source-for-dic.word.pkl"]
        ]
}

wmts = dict()
wmts["de_en"] = deen
wmts["cs_en"] = csen
wmts["fi_en"] = fien
wmts["ru_en"] = ruen
wmts["many_en"] = manyen
