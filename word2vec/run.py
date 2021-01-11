# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:53:06 2021

@author: ankur
"""

from dataset import Dataset
from word2vec import Word2Vec


# TODO(ankur): Use docopt so embedding_size, window_size, etc can be
# controlled from command line.

def run():
    ds = Dataset()
    word2vec = Word2Vec(ds, embedding_size=10)
    word2vec.sgd(40000, use_negative=True, use_saved=True)
    # word2vec.gradcheck(word2vec.skipgram_negative)
    
    
if __name__ == '__main__':
    run()