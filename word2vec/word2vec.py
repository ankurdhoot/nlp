import # -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:00:08 2021

@author: ankur
"""


class Word2Vec:
    
    def __init__(self):
        # Take in the dataset?
        # TODO(ankur): Fill this out.
        pass
    
    def skipgram(self, center_word_vec, context_word_idx, context_vectors):
        # TODO(ankur): Fill this out. Add the dimensions.
        """
        
        Arguments:
            center_word_vec -- vector representing the center word
            context_word_idx -- embedding index of the context word
            context_vectors -- the matrix of context vectors
        
        Returns:
            loss (float) -- softmax loss of context word occurring given center word
            grad_center_vec ( -- gradient w.r.t center word vector
            grad_context_vecs -- gradient w.r.t context vectors
        """
        pass