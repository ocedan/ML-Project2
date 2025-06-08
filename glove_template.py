#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


def main():
    print("loading cooccurrence matrix")
    with open("cooc.pkl", "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    bias_x = np.zeros(cooc.shape[0]) # Bias for main word vectors
    bias_y = np.zeros(cooc.shape[1]) # Bias for context word vectors

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    # GloVe weighting function
    def f(x_ij, nmax, alpha):
        return (x_ij / nmax)**alpha if x_ij < nmax else 1.0
    
    
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        # Using a fixed seed for reproducibility, remove in production if needed
        random.seed(42 + epoch) 
        indices = list(range(cooc.nnz))
        random.shuffle(indices)

        for i in indices: # Iterate through shuffled indices
            ix = cooc.row[i]
            jy = cooc.col[i]
            n = cooc.data[i]
            
            # Calculate weighting factor
            weight = f(n, nmax, alpha)

            # Use np.dot for dot product between vectors
            dot_product = np.dot(xs[ix], ys[jy])
            
            # Calculate the cost term (prediction - log(co-occurrence count))
            cost_term = dot_product + bias_x[ix] + bias_y[jy] - np.log(n)
            
            # Calculate the squared error (main part of the loss)
            # This is 2 * weight * cost_term * derivative (which is 1) for error gradient
            # The full gradient for (w_i^T * w_j_tilde) part is:
            # gradient_w_i = 2 * weight * cost_term * w_j_tilde
            # gradient_w_j_tilde = 2 * weight * cost_term * w_i
            
            # Store temporary update amounts (delta_x, delta_y)
            # These are the terms needed for updating the vectors and biases
            
            # Gradient for bias terms
            grad_bias = eta * weight * cost_term
            bias_x[ix] -= grad_bias
            bias_y[jy] -= grad_bias

            # Gradient for vectors
            # grad_x = eta * weight * cost_term * ys[jy]
            # grad_y = eta * weight * cost_term * xs[ix]
            
            # Update embeddings (simultaneously, or store temporary values)
            # Update rule: W_new = W_old - learning_rate * gradient
            
            # Temporarily store original vectors for simultaneous update
            temp_xs_ix = xs[ix].copy()
            temp_ys_jy = ys[jy].copy()

            xs[ix] -= eta * weight * cost_term * temp_ys_jy
            ys[jy] -= eta * weight * cost_term * temp_xs_ix


    # After all epochs, typically only the xs (main word embeddings) are saved
    # as they represent the final word vectors.
    np.save("embeddings.npy", xs) # Changed to .npy for clarity, typically np.save adds it.
    print("Embeddings saved to embeddings.npy")


if __name__ == "__main__":
    main()
