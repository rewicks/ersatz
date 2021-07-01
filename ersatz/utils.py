# -*- coding: utf-8 -*-

import gzip
import os
import ssl
import sys
import urllib.request
import hashlib
import shutil
import logging
import progressbar

# TODO: change the loglevel here if -q is passed
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ersatz")

USERHOME = os.path.expanduser("~")
ERSATZ_DIR = os.environ.get("ERSATZ", os.path.join(USERHOME, ".ersatz"))

MODELS = {
    "en" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/en/01.Jun.21.en.gz",
        "info" : "An English monolingual model trained on English News Commentary",
        "description" : "monolingual/en",
        "destination": "monolingual/en/01.Jun.21.en",
        "date": "01 June 2021",
        "md5" : "75e8700396a21b1fe7e08f91b1971978"
    },
    "ar" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/ar/01.Jun.21.ar.gz",
        "info" : "An Arabic monolingual model trained on Arabic News Commentary and Wikipedia data",
        "description" : "monolingual-ar",
        "destination" : "monolingual/ar/01.Jun.21.ar",
        "date" : "01 June 2021",
        "md5" : "deb6c246bd8d48478f7872668737b3e1"
    },
    "cs" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/cs/01.Jun.21.cs.gz",
        "info" : "A Czech monolingual model trained on Czech News Commentary and Wikipedia data",
        "description" : "monolingual-cs",
        "destination" : "monolingual/cs/01.Jun.21.cs",
        "date" : "01 June 2021",
        "md5" : "71fca6f2ab670843a2b698ab118b9fce"
    },
    "de" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/de/01.Jun.21.de.gz",
        "info" : "A German monolingual model trained on German News Commentary and Wikipedia data",
        "description" : "monolingual-de",
        "destination" : "monolingual/de/01.Jun.21.de",
        "date" : "01 June 2021",
        "md5" : "3dcb90a96e5a1c4e151a4bdac24db459"
    },


    "es" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/es/01.Jun.21.es.gz",
        "info" : "A Spanish monolingual model trained on Spanish News Commentary data",
        "description" : "monolingual-es",
        "destination" : "monolingual/es/01.Jun.21.es",
        "date" : "01 June 2021",
        "md5" : "6bf02a677365ead6db9efdfc7fce5586"
    },
    "et" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/et/01.Jun.21.et.gz",
        "info" : "A Estonian monolingual model trained on Estonian News Crawl data",
        "description" : "monolingual-et",
        "destination" : "monolingual/et/01.Jun.21.et",
        "date" : "01 June 2021",
        "md5" : "a8e0b2f93e4300c41097e83e34cf793f"
    },
    "fi" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/fi/01.Jun.21.fi.gz",
        "info" : "A Finnish monolingual model trained on Finnish News Crawl and Wikipedia data",
        "description" : "monolingual-fi",
        "destination" : "monolingual/fi/01.Jun.21.fi",
        "date" : "01 June 2021",
        "md5" : "02382a6b06c4e0475f4940a60fd1506c"
    },
    "fr" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/fr/01.Jun.21.fr.gz",
        "info" : "A French monolingual model trained on French News Commentary and Wikipedia data",
        "description" : "monolingual-fr",
        "destination" : "monolingual/fr/01.Jun.21.fr",
        "date" : "01 June 2021",
        "md5" : "db67663cd9bd33b4e5aad919e5c1fefa"
    },
    "gu" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/gu/01.Jun.21.gu.gz",
        "info" : "A Gujarti monolingual model trained on Gujarti News Crawl and Common Crawl data",
        "description" : "monolingual-gu",
        "destination" : "monolingual/gu/01.Jun.21.gu",
        "date" : "01 June 2021",
        "md5" : "939929a3859daafa64c8b0360bced552"
    },
    "hi" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/hi/01.Jun.21.hi.gz",
        "info" : "A Hindi monolingual model trained on Hindi News Commentary, News Crawl, and Wikipeda data",
        "description" : "monolingual-hi",
        "destination" : "monolingual/hi/01.Jun.21.hi",
        "date" : "01 June 2021",
        "md5" : "2df0baa5ce535ef2f8c221f4899ed38d"
    },
    "iu" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/iu/01.Jun.21.iu.gz",
        "info" : "A Inuktitut monolingual model trained on Inuktitut data from the Nunavut-Hansard-Inuktitut-English Parallel Corpus 3.0",
        "description" : "monolingual-iu",
        "destination" : "monolingual/iu/01.Jun.21.iu",
        "date" : "01 June 2021",
        "md5" : "c5ca9cefc5633528d039b89924030dcb"
    },
    "ja" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/ja/01.Jun.21.ja.gz",
        "info" : "A Japanese monolingual model trained on Japanese News Commentary and News Crawl data",
        "description" : "monolingual-ja",
        "destination" : "monolingual/ja/01.Jun.21.ja",
        "date" : "01 June 2021",
        "md5" : "4b6f85485757d8d5df16212193b7b2c8"
    },
    "kk" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/kk/01.Jun.21.kk.gz",
        "info" : "A Kazakh monolingual model trained on Kazakh News Commentary and News Crawl data",
        "description" : "monolingual-kk",
        "destination" : "monolingual/kk/01.Jun.21.kk",
        "date" : "01 June 2021",
        "md5" : "25464c09d8621ed3c4c0cdd594f771bb"
    },
    "km" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/km/01.Jun.21.km.gz",
        "info" : "A Khmer monolingual model trained on Khmer JW300 Corpus and Common Crawl data",
        "description" : "monolingual-km",
        "destination" : "monolingual/km/01.Jun.21.km",
        "date" : "01 June 2021",
        "md5" : "9c163b927d2641c205dcadeda8aefba4"
    },
    "lt" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/lt/01.Jun.21.lt.gz",
        "info" : "A Lithuanian monolingual model trained on Lithuanian News Crawl and Wikipedia data",
        "description" : "monolingual-lt",
        "destination" : "monolingual/lt/01.Jun.21.lt",
        "date" : "01 June 2021",
        "md5" : "b1f3ee5f2a745adf15cedcb6cb7ed2be"
    },
    "lv" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/lv/01.Jun.21.lv.gz",
        "info" : "A Latvian monolingual model trained on Latvian News Crawl data",
        "description" : "monolingual-lv",
        "destination" : "monolingual/lv/01.Jun.21.lv",
        "date" : "01 June 2021",
        "md5" : "be0c7a7d8e7f9d8933d1c4f5e1b52b0b"
    },
    "pl" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/pl/01.Jun.21.pl.gz",
        "info" : "A Polish monolingual model trained on Polish News Crawl, Global Voices, and Wikipedia data",
        "description" : "monolingual-pl",
        "destination" : "monolingual/pl/01.Jun.21.pl",
        "date" : "01 June 2021",
        "md5" : "6b874717e93147dd8f55fad2c4e7a1a3"
    },
    "ps" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/ps/01.Jun.21.ps.gz",
        "info" : "A Pashto monolingual model trained on Pashto News Crawl, SADA, SYSTRAN, and TRANSTAC data",
        "description" : "monolingual-ps",
        "destination" : "monolingual/ps/01.Jun.21.ps",
        "date" : "01 June 2021",
        "md5" : "13b2863e0d907606625743d0f091c294"
    },
    "ro" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/ro/01.Jun.21.ro.gz",
        "info" : "A Romanian monolingual model trained on Romanian News Crawl, Global Voices, and Wikipedia data",
        "description" : "monolingual-ro",
        "destination" : "monolingual/ro/01.Jun.21.ro",
        "date" : "01 June 2021",
        "md5" : "574a396c19330003b12dc91bf1e77ef5"
    },
    "ru" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/ru/01.Jun.21.ru.gz",
        "info" : "A Russian monolingual model trained on Russian News Commentary and Wikipedia data",
        "description" : "monolingual-ru",
        "destination" : "monolingual/ru/01.Jun.21.ru",
        "date" : "01 June 2021",
        "md5" : "50136fed7ad3330eb8c59c3c79864179"
    },
    "ta" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/ta/01.Jun.21.ta.gz",
        "info" : "A Tamil monolingual model trained on Tamil Wikipedia and News Crawl data",
        "description" : "monolingual-ta",
        "destination" : "monolingual/ta/01.Jun.21.ta",
        "date" : "01 June 2021",
        "md5" : "00785d88c84ee656343b0d12082e1c3a"
    }, 
    "tr" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/tr/01.Jun.21.tr.gz",
        "info" : "A Turkish monolingual model trained on Turkish Global Voices, Wikipedia, and News Crawl data",
        "description" : "monolingual-tr",
        "destination" : "monolingual/tr/01.Jun.21.tr",
        "date" : "01 June 2021",
        "md5" : "5702b95c97d9702fef7ab57380a80a1d"
    },
    "zh" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/zh/01.Jun.21.zh.gz",
        "info" : "A Chinese monolingual model trained on Chinese News Commentary and Wikipedia data",
        "description" : "monolingual-zh",
        "destination" : "monolingual/zh/01.Jun.21.zh",
        "date" : "01 June 2021",
        "md5" : "385e068e30f54963fbc582f49f4416ff"
    },
    "default-multilingual" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/multilingual/wmtlangs/01.Jun.21.multilingual.gz",
        "info": "A multilingual model, including languages commonly associated with WMT tasks and datasets",
        "description" : "multilingual/wmtlangs",
        "destination": "multilingual/wmtlangs/01.Jun.21.multilingual",
        "date": "01 June 2021",
        "md5" : "2d7d2092800cecda2b88f9da9fffbfff"
    }
}

def list_models():
    for model_name in MODELS:
        model = MODELS[model_name]
        print(f'\t- {model_name} ({model["description"]}) : {model["info"]}')
    pass

def get_model_path(model_name='default-multilingual'):

    if model_name not in MODELS:
        logger.error(f"Could not find model by name of \"{model_name}\". Using \"default-multilingual\" instead")
        model_name = 'default-multilingual'

    model = MODELS[model_name]

    logger.info(f"Segmentation model: \"{model_name}\"")
    logger.info(f"Model description: \"{model['description']}\"")
    logger.info(f"Release Date: \"{model['date']}\"")

    model_file = os.path.join(ERSATZ_DIR, model['destination'])
    if os.path.exists(model_file):
        logger.info(f"USING \"{model_name}\" model found at {model_file}")
        return model_file
    elif download_model(model_name) == 0:
        return model_file
    sys.exit(1)

pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download_model(model_name='default'):
    """
    Downloads the specified model into the ERSATZ directory
    :param language:
    :return:
    """

    expected_checksum = MODELS[model_name].get('md5', None)
    model_source = MODELS[model_name]['source']
    model_file = os.path.join(ERSATZ_DIR, os.path.basename(model_source))
    model_destination = os.path.join(ERSATZ_DIR, MODELS[model_name]['destination'])

    os.makedirs(ERSATZ_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(model_destination), exist_ok=True)

    logger.info(f"DOWNLOADING \"{model_name}\" model from {model_source}")

    if not os.path.exists(model_file) or os.path.getsize(model_file) == 0:
        try:
            urllib.request.urlretrieve(model_source, model_file, show_progress)
        except Exception as e:
            logger.error(e)
            sys.exit(1)

    if expected_checksum is not None:
        md5 = hashlib.md5()
        with open(model_file, 'rb') as infile:
            for line in infile:
                md5.update(line)
        if md5.hexdigest() != expected_checksum:
            logger.error(f"Failed checksum: expected was {expected_checksum}, received {md5.hexdigest()}")
            sys.exit(1)

        logger.info(f"Checksum passed: {md5.hexdigest()}")

    logger.info(f"EXTRACTING {model_file} to {model_destination}")
    with gzip.open(model_file) as infile, open(model_destination, 'wb') as outfile:
        shutil.copyfileobj(infile, outfile)

    return 0
