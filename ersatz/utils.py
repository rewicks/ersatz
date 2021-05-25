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
    stream=sys.stdout,
)
logger = logging.getLogger("ersatz")

USERHOME = os.path.expanduser("~")
ERSATZ_DIR = os.environ.get("ERSATZ", os.path.join(USERHOME, ".ersatz"))

MODELS = {
    "en" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/monolingual/en/01.Jun.21.en.gz",
        "description" : "monolingual/en",
        "destination": "monolingual/en/01.Jun.21.en",
        "date": "01 June 2021"
    },
    "default_multilingual" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/multilingual/wmtlangs/01.Jun.21.multilingual.gz",
        "description" : "multilingual/wmtlangs/",
        "destination": "multilingual/wmtlangs/01.Jun.21.multilingual",
        "date": "01 June 2021"
    }
}

def get_model_path(model_name='default_multilingual'):

    if model_name not in MODELS:
        logger.error(f"Could not find model by name of \"{model_name}\". Using \"default_multilingual\" instead")
        model_name = 'default_multilingual'

    model = MODELS[model_name]

    logger.info(f"Segmentation model: \"{model_name}\"")
    logger.info(f"Model description: \"{model['description']}\"")
    logger.info(f"Release Date: \"{model['date']}\"")

    model_source = model['source']
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
