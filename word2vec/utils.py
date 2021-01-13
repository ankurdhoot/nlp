# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:24:36 2021

@author: ankur
"""
import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))

    return s

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 
    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x

def cosine_similarity(vocab_embeddings, word_embedding):
    """ 
    Returns the cosine similarity between the word_embedding and all
    vocab_embeddings.
    
    Arguments:
        vocab_embeddings (vocab_size, embedding_size) -- the vocab embeddings
        word_embedding (embedding_size, ) -- the word embedding
    """
    
    # (vocab_size, )
    dot_product = np.dot(vocab_embeddings, word_embedding)
    
    # (vocab_size, )
    vocab_embedding_norm = np.linalg.norm(vocab_embeddings, axis=1)
    
    # ()
    word_embedding_norm = np.linalg.norm(word_embedding)
    
    # (vocab_size, )
    cosine_similarity = dot_product / (vocab_embedding_norm * word_embedding_norm)
    
    return cosine_similarity