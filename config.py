#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:32:48 2021

@author: user1
"""

import transformers

# =============================================================================
# MAX_LEN = 128
# TRAIN_BATCH_SIZE = 32
# VALID_BATCH_SIZE = 8
# EPOCHES = 10
# BASE_MODEL_PATH = '../input/bert_base_uncased'
# MODEL_PATH = ''
# TAINING_FILE = '../input/ner_dataset.csv'
# 
# TOKENIZER = transformers.BertTokenizer.from_pretrained(
#         BASE_MODEL_PATH,
#         do_lower_case =  True)
# =============================================================================


MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 1
BASE_MODEL_PATH = "/home/user1/deskdata/News_IE/BERT_LEARN/input/bert_base_uncased"
MODEL_PATH = "/home/user1/deskdata/News_IE/BERT_LEARN/model.bin"
TRAINING_FILE = "/home/user1/deskdata/News_IE/BERT_LEARN/input/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
