# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:12:52 2021

@author: ankur
"""

import random
from dataset import Dataset

def test_get_context():
    ds = Dataset()
    center_word_id, context_word_ids = ds.get_context()
    id2token = ds.id2token()
    print(id2token[center_word_id])
    for context_word_id in context_word_ids:
        print(id2token[context_word_id])
    
    
def test_num_tokens():
    ds = Dataset()
    assert ds.num_tokens() == 19538
    
    
    
if __name__ == '__main__':
    test_get_context()
    test_num_tokens()