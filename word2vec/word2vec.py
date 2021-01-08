# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:00:08 2021

@author: ankur
"""

import numpy as np
from utils import softmax, sigmoid
import time

class Word2Vec:
    
    # TODO(ankur): Experiment with different window sizes.
    
    def __init__(self, dataset, embedding_size=10):
        """ 
        Arguments:
            dataset -- the dataset on which to learn embeddings
            embedding_size -- the dimensionality of the embeddings to learn
        """
        self.dataset = dataset
        self.vocab_size = dataset.num_tokens()
        self.embedding_size = embedding_size
        # Initialize to [-0.5, 0.5)
        # (vocab_size, embedding_size)
        self.center_word_vectors = np.random.rand(self.vocab_size, embedding_size) - 0.5
        # (vocab_size, embedding_size)
        self.context_word_vectors = np.random.rand(self.vocab_size, embedding_size) - 0.5
    
    def skipgram_loss_and_gradient(self, center_word_idx, context_word_idx):
        """
        
        Arguments:
            center_word_idx (int) -- embedding index of the center word
            context_word_idx (int) -- embedding index of the context word
        
        Returns:
            loss (float) -- softmax loss of context word occurring given center word
            grad_center_vec (embedding_size, ) -- gradient w.r.t center word vector
            grad_context_vecs (vocab_size, embedding_size) -- gradient w.r.t context vectors
        """
        assert center_word_idx < self.vocab_size
        assert context_word_idx < self.vocab_size
        
        # (embedding_size, )
        center_word_vec = self.center_word_vectors[center_word_idx]
        
        # (vocab_size, )
        logits = np.matmul(self.context_word_vectors, center_word_vec)
        
        # (vocab_size, )
        probs = softmax(logits)
        
        # Cross entropy loss
        loss = -np.log(probs[context_word_idx])
        
        # Calculate the gradients
        # Let W = self.center_word_vectors
        # Let C = self.context_word_vectors
        # Let J(x, y) be the loss on context word x given center word y
        
        # (embedding_size, )
        dJ_dWy = np.zeros(self.embedding_size)
        # (vocab_size, embedding_size)
        dJ_dC = np.zeros(self.context_word_vectors.shape)
        
        # (vocab_size, )
        y_hat = probs
        
        # (vocab_size, )
        y = np.zeros(self.vocab_size)
        y[context_word_idx] = 1
        
        # (vocab_size, )
        delta = y_hat - y
        
        # C.T * (y_hat - y)
        # (embedding_size, vocab_size) * (vocab_size, ) --> (embedding_size, )
        dJ_dWy = np.matmul(np.transpose(self.context_word_vectors), delta)
        
        # (y_hat - y) * W_y
        # (vocab_size, ) . (embedding_size, ) --> (vocab_size, embedding_size)
        dJ_dC = np.outer(delta, center_word_vec)
        
        return loss, dJ_dWy, dJ_dC
    
    
    def skipgram(self):
        """
        Samples one word from the dataset and retrieves its context words.
        Computes the loss and gradients on this single example. 
        
        Returns:
            loss(float) -- softmax loss
            grad_center_vecs -- gradient w.r.t center word vectors
            grad_context_vecs -- gradient w.r.t outside word vectors

        """
        
        loss = 0
        
        # Let W = self.center_word_vectors
        # Let C = self.context_word_vectors
        # Let J(x, y) be the loss on context word x given center word y
        dJ_dW = np.zeros(self.center_word_vectors.shape)
        dJ_dC = np.zeros(self.context_word_vectors.shape)
        
        center_word_id, context_word_ids = self.dataset.get_context()
        
        for context_word_id in context_word_ids:
            # int, (embedding_size, ), (vocab_size, embedding_size)
            loss_word, grad_center_word_vec, grad_context_word_vecs = self.skipgram_loss_and_gradient(center_word_id, context_word_id)
            # Acculate the loss and gradients
            loss += loss_word
            dJ_dW[center_word_id] += grad_center_word_vec
            dJ_dC += grad_context_word_vecs
            
        return loss, dJ_dW, dJ_dC
    
    def skipgram_batch(self, batchsize = 50):
        """ 
        Arguments:
            batchsize (int) -- The number of examples to average over
            
        Returns:
            loss (float) -- softmax loss
            grad_center_vecs -- gradient w.r.t center word vectors
            grad_context_vecs -- gradient w.r.t outside word vectors
        """
        
        # average loss over all examples
        batch_loss = 0
        # gradient w.r.t center_word_vectors
        batch_dJ_dW = np.zeros(self.center_word_vectors.shape)
        # gradient w.r.t context_word_vectors
        batch_dJ_dC = np.zeros(self.context_word_vectors.shape)
        
        for _ in range(batchsize):
            loss, dJ_dW, dJ_dC = self.skipgram()
            batch_loss += loss / batchsize
            batch_dJ_dW += dJ_dW / batchsize
            batch_dJ_dC += dJ_dC / batchsize
            
        return batch_loss, batch_dJ_dW, batch_dJ_dC
    
    def skipgram_negative_loss_and_gradient(self, center_word_id, context_word_id, negative_sample_ids):
        """
        
        Arguments:
            center_word_id (int) -- embedding index of the center word
            context_word_id (int) -- embedding index of the context word
            negative_sample_ids (List[int]) -- embedding indices of the negatively sample words
        
        Returns:
            loss (float) -- softmax loss of context word occurring given center word
            grad_center_vec (embedding_size, ) -- gradient w.r.t center word vector
            grad_context_vecs (vocab_size, embedding_size) -- gradient w.r.t context vectors
        """
        assert center_word_idx < self.vocab_size
        assert context_word_idx < self.vocab_size
        
        for negative_sample_id in negative_sample_ids:
            assert negative_sample_id < self.vocab_size
            
        # (embedding_size, )
        center_word_vec = self.center_word_vectors[center_word_idx]
        
        # (K + 1) entries
        context_and_negative_ids = [context_word_id] + negative_sample_ids
        
        # The entry corresponding to the context word is 1. The entries 
        # corresponding to negative words are -1. This setup aids in the
        # loss and gradient calculations.
        labels = np.array([1] + [-1 for _ in range(len(negative_sample_ids))])
        
        # (K + 1, embedding_size)
        context_and_negative_vecs = self.context_word_vectors[labels, :]
        
        # (K + 1, embedding_size) * (embedding_size, ) --> (K + 1, )
        logits = np.matmul(self.context_and_negative_vecs, center_word_vec)
        
        # (K + 1, )
        probs = sigmoid(logits * labels)
        
        loss = -np.sum(np.log(probs))
                
        # Calculate the gradients
        # Let W = self.center_word_vectors
        # Let C = self.context_word_vectors
        # Let J(x, y) be the loss on context word x given center word y
            
        # (embedding_size, )
        dJ_dWy = np.zeros(self.embedding_size)
        # (vocab_size, embedding_size)
        dJ_dC = np.zeros(self.context_word_vectors.shape)
        
        # (K+1, embedding_size)
        grad_context_and_negative_sample_vecs = np.zeros(context_and_negative_vecs.shape)
        
        # sum((K + 1) * (K + 1) * (K + 1, embedding_size), axis=0) --> 
        # sum((K + 1, embedding_size), axis=0) --> (embedding_size, )
        dJ_dWy = np.sum((probs - 1) * labels * context_and_negative_vecs, axis=0)
        
        # np.outer((K + 1), (embedding_size, )) --> (K + 1, embedding_size)
        grad_context_and_negative_sample_vecs = np.outer((probs - 1) * labels, center_word_vec)
        
        for k, gradient in grad_context_and_negative_sample_vecs:
            dJ_dC[context_and_negative_ids[k]] += gradient
            
            
        return loss, dJ_dWy, dJ_dC
        
    def skipgram_negative(self):
        """
        Samples one word from the dataset and retrieves its context words
        along with their respective negative samples. 
        
        Returns:
            loss(float) -- softmax loss
            grad_center_vecs -- gradient w.r.t center word vectors
            grad_context_vecs -- gradient w.r.t outside word vectors

        """
        
        loss = 0
        # Let W = self.center_word_vectors
        # Let C = self.context_word_vectors
        # Let J(x, y) be the loss on context word x given center word y
        dJ_dW = np.zeros(self.center_word_vectors.shape)
        dJ_dC = np.zeros(self.context_word_vectors.shape)
        
        center_word_id, context_word_ids = self.dataset.get_context()
        
        for context_word_id in context_word_ids:
            negative_sample_ids = self.dataset.get_negative_samples(context_word_id)
            # int, (embedding_size, ), (vocab_size, embedding_size)
            loss_word, grad_center_word_vec, grad_context_word_vecs = self.skipgram_negative_loss_and_gradient(center_word_id, context_word_id, negative_sample_ids)
            
            # Acculate the loss and gradients
            loss += loss_word
            dJ_dW[center_word_id] += grad_center_word_vec
            dJ_dC += grad_context_word_vecs
        
        return loss, dJ_dW, dJ_dC
    
    def skipgram_negative_batch(self, batchsize = 50):
        """ 
        Arguments:
            batchsize (int) -- The number of examples to average over
            
        Returns:
            loss (float) -- softmax loss
            grad_center_vecs -- gradient w.r.t center word vectors
            grad_context_vecs -- gradient w.r.t outside word vectors
        """
        
        # TODO(ankur): Refactor with skipgram_batch()
        
        # average loss over all examples
        batch_loss = 0
        # gradient w.r.t center_word_vectors
        batch_dJ_dW = np.zeros(self.center_word_vectors.shape)
        # gradient w.r.t context_word_vectors
        batch_dJ_dC = np.zeros(self.context_word_vectors.shape)
        
        for _ in range(batchsize):
            loss, dJ_dW, dJ_dC = self.skipgram()
            batch_loss += loss / batchsize
            batch_dJ_dW += dJ_dW / batchsize
            batch_dJ_dC += dJ_dC / batchsize
            
        return batch_loss, batch_dJ_dW, batch_dJ_dC    
            
        
    
    def sgd(self, skipgram_function, iterations):
        """ 
        Arguments:
            skipgram_function (function) -- the variant of skipgram to use
            iterations (int) -- the number of iterations to run SGD
        """
        # The learning rate.
        lr = 0.3
        
        # How often to decrease the learning rate
        ANNEAL_EVERY = 10000
        
        # How often to output statistics.
        LOG_EVERY = 10
        
        # Each iteration uses this many examples
        BATCHSIZE = 50
        
        # cumulative exponential loss
        exploss = None
        
        # Track the total time taken so far
        begin_time = time.time()
        
        for iteration in range(iterations):
            
            iter_train_time = time.time()
            
            loss, grad_center_vecs, grad_context_vecs = self.skipgram_batch(BATCHSIZE)
            
            # Gradient update
            self.center_word_vectors = self.center_word_vectors - lr * grad_center_vecs
            self.context_word_vectors = self.context_word_vectors - lr * grad_context_vecs
            
            
            if iteration % LOG_EVERY == 0:
                if not exploss:
                    exploss = loss
                else:
                    exploss = 0.95 * exploss + .05 * loss
                print("iter %d, exploss %f,  loss %f speed (examples/sec) %.2f, time elapsed %.2f" \
                      % (iteration, exploss, loss, BATCHSIZE / (time.time() - iter_train_time), time.time() - begin_time))
                    
            if iteration % ANNEAL_EVERY == 0:
                lr = lr / 2
        
    
    def run(self):
        # sgd = SGD()
        # dataset = Dataset()
        # while (loss is improving):
        #     total_loss = 0
        #     total_grad_center_vecs = 0
        #     total_grad_context_vecs = 0
        #     for i in range (batchsize):
        #         loss, grad_center_vecs, grad_context_vecs = skipgram();
        #         total_loss += loss
        #         total_grad_center_vecs += grad_center_vecs
        #         total_grad_context_vecs += grad_context_vecs
        #         loss = loss / batchsize
        pass
