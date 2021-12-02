#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:18:25 2021

@author: user1
"""

import numpy as np
import joblib
import torch
from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel



if __name__ == '__main__':

    
    meta_data = joblib.load('meta.bin')
    
    enc_pos = meta_data['enc_pos']
    enc_tag = meta_data['enc_tag']
    
    
    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))
    
    sentence = 'honggen is going to china'
    print(sentence)
    
    tokenized_sentence = config.TOKENIZER.encode(sentence)
    
    sentence = sentence.split()
    
    test_dataset = dataset.EntityDataset(
        texts = [sentence],
        pos = [[0]*len(sentence)],
        tags = [[0]*len(sentence)])
    


    
    device = torch.device("cpu")
    model = EntityModel(num_tag = num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH), map_location=device)
    model.to(device)
    
    
    with torch.no_grad():
        data = test_dataset[0]
        
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos,_ = model(**data)
        
        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
        print(
            enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
        
        
            
    
    
    


