# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:04:39 2021

@author: ankur
"""

import numpy as np
import random

class Dataset:
    
    # TODO(ankur): Implement subsampling.
    
    def __init__(self):
        # TODO(ankur): Fill this out
        self.dataset_path = "datasetSentences.txt"
        pass
    
    def sentences(self):
        """
        Stores the list of sentences from the dataset.
        Returns:
            self._sentences (List[List[str]]) - the sentences

        """
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences
        
        sentences = []
        with open(self.dataset_path, 'r') as f:
            # Skip the first line
            f.readline()
            for line in f:
                # The first token is the line number
                splitted = line.strip().split()[1:]
                sentences += [[word.lower() for word in splitted]]
        
        self._sentences = sentences
        return self._sentences
        
    def num_sentences(self):
        return len(self.sentences())
    
    def tokens(self):
        """
        Compute and store a mapping of token -> id.
        
        Returns:
            self._token2id (dict) -- token to id mapping
        """
        if hasattr(self, "_token2id") and self._token2id:
            return self._token2id
        
        token2id = dict()
        id2token = dict()
        tokenfreq = dict();
        idx = 0
        
        for sentence in self.sentences():
            for token in sentence:
                if not token in token2id:
                    token2id[token] = idx
                    id2token[idx] = token
                    tokenfreq[token] = 1
                    idx += 1
                else:
                    tokenfreq[token] += 1
        
        # TODO(ankur): Do I need to do anything about unknown words?
        self._id2token = id2token
        self._token2id = token2id
        
        return self._token2id
    
    def id2token(self):
        # Compute the dicts if not already done.
        self.tokens()
        assert hasattr(self, "_id2token") and self._id2token
        return self._id2token
    
    def num_tokens(self):
        return len(self.tokens())
        
    def get_context(self, C=5):
        """
        Returns a random center word along with its context words.
        Arguments:
            C (int) -- the (one-sided) size of the training context
        
        Return:
            (int, List[int]) : indices of the center word along with 2*C context words

        """
        
        sent_num = random.randint(0, self.num_sentences() - 1)
        sentence = self.sentences()[sent_num]
        center_word_num = random.randint(0, len(sentence) - 1)
        center_word = sentence[center_word_num]
        
        # Get the C before and C after context words
        context_words = []
        context_words += sentence[max(0, center_word_num - C):center_word_num]
        context_words += sentence[center_word_num + 1: center_word_num + C + 1]
        
        center_word_id = self.tokens()[center_word]
        context_word_ids = [self.tokens()[context_word] for context_word in context_words]
        
        if len(context_word_ids) == 0:
            return self.get_context()
        
        return center_word_id, context_word_ids
        
        
        
        
        
        
        
