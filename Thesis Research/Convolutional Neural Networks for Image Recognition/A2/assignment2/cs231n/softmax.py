import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  #print('num_classes = ', num_classes)
  num_train = X.shape[0]
  #print('num_train = ', num_train)
  
  min_score = 0.0
  shifted_scores = np.zeros(W.shape[1])
  #max_score = np.zeros(W.shape[1])
  max_score = 0.0
  
  loss_array = np.zeros(y.shape[0])
  for i in range(num_train):
    scores = X[i].dot(W)
    #print('scores dimensions = ', scores.shape)
    #print('scores = ', scores)
    #print('i =', i, 'y = ', y[i])
    min_score = np.min(scores)
    max_score = np.max(scores)
    #print(min_score,max_score)
    shifted_scores = np.multiply(-1,scores + abs(min_score))
    #print(scores)
    #print(shifted_scores)
    exp_scores = np.exp(shifted_scores)
    norm = np.amax(exp_scores)
    norm_scores = np.divide(exp_scores,norm)
    loss_array[i] = np.multiply(-1,np.log(norm_scores[y[i]]/(np.sum(norm_scores)-norm_scores[y[i]])))
    #print(loss_array)
    for j in range(num_classes): 
	
        if j == y[i]: 
            dW[:,j] = np.multiply(norm_scores[y[i]],1-norm_scores[y[i]])
        else:
            dW[:,j] = np.multiply(-1,np.multiply(norm_scores[y[i]],norm_scores[y[j]]))
			
			
  loss = np.amax(loss_array)

  # Add regularization to the loss.
  loss = 0.5 * reg * np.sum(W * W) + loss
  
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

