# -*- coding: utf-8 -*-

import gzip
import os
import ssl
import sys
import urllib.request
import hashlib
import shutil

USERHOME = os.path.expanduser("~")
ERSATZ_DIR = os.environ.get("ERSATZ", os.path.join(USERHOME, ".ersatz"))

MODELS = {
    "en" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/ersatz.en.gz",
        "description" : "English Monolingual Model"
    },
    "default" : {
        "source" : "https://github.com/rewicks/ersatz-models/raw/main/ersatz.multilingual.gz",
        "description" : "A multilingual model"
    }
}

def get_model_path(model_name='default'):
    model_source = MODELS[model_name]['source']
    model_file = os.path.join(ERSATZ_DIR, os.path.basename(model_source)).replace('.gz', '')
    if os.path.exists(model_file):
        return model_file
    elif download_model(model_name) == 0:
        return model_file
    sys.exit(1)

def download_model(model_name='default'):
    """
    Downloads the specified model into the ERSATZ directory
    :param language:
    :return:
    """
    os.makedirs(ERSATZ_DIR, exist_ok=True)

    expected_checksum = MODELS[model_name].get('md5', None)
    model_source = MODELS[model_name]['source']
    model_file = os.path.join(ERSATZ_DIR, os.path.basename(model_source))

    if not os.path.exists(model_file) or os.path.getsize(model_file) == 0:
        try:
            with urllib.request.urlopen(model_source) as f, open(model_file, 'wb') as out:
                out.write(f.read())
        except ssl.SSLError:
            sys.exit(1)

    if expected_checksum is not None:
        md5 = hashlib.md5()
        with open(model_file, 'rb') as infile:
            for line in infile:
                md5.update(line)
        if md5.hexdigest() != expected_checksum:
            sys.exit(1)
        else:
            pass

    with gzip.open(model_file) as infile, open(model_file.replace('.gz', ''), 'wb') as outfile:
        shutil.copyfileobj(infile, outfile)

    return 0
