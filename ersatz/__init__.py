#!/usr/bin/env python
# -*- coding: utf-8 -*-


__version__ = '1.0.0'
__description__ = 'Simple sentence segmentation toolkit for segmenting and scoring'

class DummyArgs():
    def __init__(self):
        return

def split(model="default-multilingual",
          text=None,
          input=None,
          output=None,
          batch_size=16,
          candidates="multilingual",
          cpu=False,
          columns=None,
          delimiter='\t'):
    from .split import split as ersatz_split
    args = DummyArgs()
    args.model = model
    args.text = text
    args.input = input
    args.output = output
    args.batch_size = batch_size
    args.candidates = candidates
    args.cpu = cpu
    args.columns = columns
    args.delimiter = delimiter
    args.list = True
    return ersatz_split(args)


def train():
    raise NotImplementedError

def score():
    raise NotImplementedError
