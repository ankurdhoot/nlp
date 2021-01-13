# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:00:08 2021

@author: ankur
"""

import numpy as np
from utils import softmax, sigmoid, cosine_similarity
import time
import glob
import os.path as op
import random

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
        assert center_word_id < self.vocab_size
        assert context_word_id < self.vocab_size
        
        for negative_sample_id in negative_sample_ids:
            assert negative_sample_id < self.vocab_size
            
        # (embedding_size, )
        center_word_vec = self.center_word_vectors[center_word_id]
        
        # (K + 1) entries, the context_word_id comes first
        context_and_negative_ids = np.insert(negative_sample_ids, 0, context_word_id)
        
        # The entry corresponding to the context word is 1. The entries 
        # corresponding to negative words are -1. This setup aids in the
        # loss and gradient calculations.
        labels = np.array([1] + [-1 for _ in range(len(negative_sample_ids))])
        
        # (K + 1, embedding_size)
        context_and_negative_vecs = self.context_word_vectors[labels, :]
        
        # (K + 1, embedding_size) * (embedding_size, ) --> (K + 1, )
        logits = np.matmul(context_and_negative_vecs, center_word_vec)
        
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
        
      
        # np.dot((embedding_size, K + 1), (K + 1)) --> (embedding_size, )
        dJ_dWy = np.dot(context_and_negative_vecs.T, (probs - 1) * labels)
        
        assert dJ_dWy.shape == (self.embedding_size, )
        
        # sum((K + 1) * (K + 1) * (K + 1, embedding_size), axis=0) --> 
        # sum((K + 1, embedding_size), axis=0) --> (embedding_size, )
        # dJ_dWy = np.sum((probs - 1) * labels * context_and_negative_vecs, axis=0)
        
        # np.outer((K + 1), (embedding_size, )) --> (K + 1, embedding_size)
        grad_context_and_negative_sample_vecs = np.outer((probs - 1) * labels, center_word_vec)
        
        for k, gradient in enumerate(grad_context_and_negative_sample_vecs):
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
            loss, dJ_dW, dJ_dC = self.skipgram_negative()
            batch_loss += loss / batchsize
            batch_dJ_dW += dJ_dW / batchsize
            batch_dJ_dC += dJ_dC / batchsize
            
        return batch_loss, batch_dJ_dW, batch_dJ_dC    
            
    def load_saved_params(self, use_negative=False):
        """ 
        Load the saved parameters if they exist.
        Arguments:
            use_negative (bool) -- whether the parameters are from negative sampling
        Returns:
            max_iter (int) -- the latest iteration saved
        """
        # Find the latest saved set of parameters
        max_iter = 0
        
        center_params_file = None
        context_params_file = None
        glob_file = None
        
        if use_negative:
            glob_file = "neg_saved_center_params_*.npy"
            center_params_file = "neg_saved_center_params_%d.npy"
            context_params_file = "neg_saved_context_params_%d.npy"
        else:
            glob_file = "saved_center_params_*.npy"
            center_params_file = "saved_center_params_%d.npy"
            context_params_file = "saved_context_params_%d.npy"
            
        for f in glob.glob(glob_file):
            if use_negative:
                iteration = int(op.splitext(op.basename(f))[0].split("_")[4])
            else:
                iteration = int(op.splitext(op.basename(f))[0].split("_")[3])
                
            max_iter = max(iteration, max_iter)
        
        center_word_vecs = None
        context_word_vecs = None
        
        if max_iter > 0:
            center_params_file = center_params_file % max_iter
            context_params_file = context_params_file % max_iter
            
            self.center_word_vectors = np.load(center_params_file)
            self.context_word_vectors = np.load(context_params_file)
                
            
        return max_iter
            
        
        
    def save_params(self, iteration, use_negative=False):
        """  
        Arguments:
            iteration (int) -- the iteration number during SGD
            use_negative (bool) -- whether the parameters are from negative sampling
        """
        
        center_params_file = None
        context_params_file = None
        
        if use_negative:
            center_params_file = "neg_saved_center_params_%d.npy"
            context_params_file = "neg_saved_context_params_%d.npy"
        else:
            center_params_file = "saved_center_params_%d.npy"
            context_params_file = "saved_context_params_%d.npy"
            
        center_params_file = center_params_file % iteration
        context_params_file = context_params_file % iteration
        np.save(center_params_file, self.center_word_vectors)
        np.save(context_params_file, self.context_word_vectors)
        
        
    
    def sgd(self, iterations, use_negative=False, use_saved=False):
        """ 
        Run SGD and optimize the embeddings.
        Arguments:
            iterations (int) -- the number of iterations to run SGD
            use_negative (bool) -- whether to use the negative sampling method
            use_saved (bool) -- whether to use the latest saved parameters
        Returns:
            center_word_vectors (vocab_size, embedding_size)
            context_word_vectors (vocab_size, embedding_size)
        """
        # The learning rate.
        lr = 0.3
        
        # How often to decrease the learning rate
        ANNEAL_EVERY = 10000
        
        # How often to output statistics.
        LOG_EVERY = 10
        
        # How often to save the parameters
        SAVE_PARAMS_EVERY = ANNEAL_EVERY
        
        # Each iteration uses this many examples
        BATCHSIZE = 50
        
        # Load the saved parameters if requested
        
        if use_saved:
            max_iter = self.load_saved_params(use_negative)
                
            # Adjust the learning rate based on the number of iterations
            lr = lr * (.5) ** (max_iter / ANNEAL_EVERY)
        
        # cumulative exponential loss
        exploss = None
        
        # the skipgram variant to use
        skipgram_function = None
        if use_negative:
            skipgram_function = self.skipgram_negative_batch
        else:
            skipgram_function = self.skipgram_batch
        
        # Track the total time taken so far
        begin_time = time.time()
        
        for iteration in range(iterations):
            
            iter_train_time = time.time()
            
            loss, grad_center_vecs, grad_context_vecs = skipgram_function(BATCHSIZE)
            
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
                
            if iteration % SAVE_PARAMS_EVERY == 0:
                print("--------------------")
                print("Saving parameters")
                self.save_params(iteration, use_negative)
        
        return self.center_word_vectors, self.context_word_vectors
        
    
    def gradcheck(self, skipgram_function):
        """
        Run gradient checking.
        
        Arguments:
            skipgram_function (function) -- the variant of skipgram to use for gradient checking
        """
        
        # Change this to evaluate the gradients at a different point.
        np.random.seed(10)
        
        # amount by which we nudge the point
        delta_x = 1e-7
        
        # the tolerable relative difference between approximate and analytical gradient.
        epsilon = 1e-5
        
        # Let's evaluate the gradients at a random point.
        # Initialize to [-0.5, 0.5)
        # (vocab_size, embedding_size)
        self.center_word_vectors = np.random.rand(self.vocab_size, self.embedding_size) - 0.5
        # (vocab_size, embedding_size)
        self.context_word_vectors = np.random.rand(self.vocab_size, self.embedding_size) - 0.5
        
        # We need to always use the same dataset points when evaluating the function
        state = random.getstate()
        random.setstate(state)
        
        # Evaluate the funtion at the original point
        loss, grad_center_vecs, grad_context_vecs = skipgram_function()
        
        gradient_check_successful = True
        
        # First check the gradients for center word vectors.
        with np.nditer(self.center_word_vectors, flags=['multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                
                # Reset
                random.setstate(state)
                
                # Compute f(x + delta_x)
                self.center_word_vectors[it.multi_index] += delta_x
                loss_plus, grad_center_vecs_plus, grad_context_vecs_plus = skipgram_function()
                
                
                # Reset
                random.setstate(state)
                
                # Compute f(x - delta_x)
                self.center_word_vectors[it.multi_index] -= 2 * delta_x
                loss_neg, grad_center_vecs_neg, grad_context_vecs_neg = skipgram_function()
                
                # Reset center_word_vectors to original value
                self.center_word_vectors[it.multi_index] += delta_x
                
                grad_approx = (loss_plus - loss_neg) / (2 * delta_x)
                grad_real = grad_center_vecs[it.multi_index]
                
                
                rel_diff = abs(grad_approx - grad_real) / max(abs(grad_approx), abs(grad_real), 1)
                
                if rel_diff > epsilon:
                    gradient_check_successful = False
                    print("Gradient check failed")
        
        # Now check the gradients for the context word vectors.
        with np.nditer(self.context_word_vectors, flags=['multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                
                # Reset
                random.setstate(state)
                
                # Compute f(x + delta_x)
                self.context_word_vectors[it.multi_index] += delta_x
                loss_plus, grad_center_vecs_plus, grad_context_vecs_plus = skipgram_function()
                
                
                # Reset
                random.setstate(state)
                
                # Compute f(x - delta_x)
                self.context_word_vectors[it.multi_index] -= 2 * delta_x
                loss_neg, grad_center_vecs_neg, grad_context_vecs_neg = skipgram_function()
                
                # Reset center_word_vectors to original value
                self.center_word_vectors[it.multi_index] += delta_x
                
                grad_approx = (loss_plus - loss_neg) / (2 * delta_x)
                grad_real = grad_context_vecs[it.multi_index]
                
                
                rel_diff = abs(grad_approx - grad_real) / max(abs(grad_approx), abs(grad_real), 1)
                
                if rel_diff > epsilon:
                    gradient_check_successful = False
                    print("Gradient check failed")
                    
        if gradient_check_successful:
            print("Gradient check passed!")
            
            
    def k_nearest_words(self, word, k=3):
        """
        Arguments:
            word (str): The word for which the closest word is requested.
            k (int): The number of nearest words to find.

        """
        self.load_saved_params(use_negative=True)
        
        if not word in self.dataset.token2id():
            return None
        
        word_id = self.dataset.token2id()[word]
        
        # (embedding_size, )
        word_embedding = self.center_word_vectors[word_id]
        
        similarity = cosine_similarity(self.center_word_vectors, word_embedding)
        
        top_k_ind = np.argpartition(similarity, -k)[-k:]
        
        print(word)
        
        for word_idx in top_k_ind:
            similar_word = self.dataset.id2token()[word_idx]
            similarity_score = similarity[word_idx]
            print("%s : %f" % (similar_word, similarity_score))
            
    def k_nearest_words_random(self, k=15):
        random_word = self.dataset.id2token()[random.randint(0, len(self.dataset.id2token()))]
        return self.k_nearest_words(random_word, k)
        
                
                
                
                
                
                
        
