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
  
  P = np.zeros([num_classes,y.shape[0]])
  Py = np.empty([y.shape[0]])
  loss_array = y*0.0
  exp_scores = np.zeros([num_classes,y.shape[0]])
  
  pos_max = np.amax(X)
  neg_min = np.amin(X)
  
  normalized_input = X
  
  for i in range (X.shape[0]): 
        for j in range (X.shape[1]): 
            if X[i,j] > 0: 
                normalized_input[i,j] = X[i,j] / pos_max
            elif X[i,j] < 0: 
                normalized_input[i,j] = X[i,j] / abs(neg_min)
  #print('normalized_input = ', normalized_input)
  
  #print(np.amax(normalized_input))
  #print(np.amin(normalized_input))
  
  for i in range(num_train):
    scores = normalized_input[i,:].dot(W)
    #print('scores dimensions = ', scores.shape)
    #print('scores = ', scores)
    #print('i =', i, 'y = ', y[i])

    exp_scores[:,i] = np.exp(scores)
    #print('exp_scores shape = ', exp_scores.shape)
    sum_exp_scores = np.sum(exp_scores[:,i])
	
    P[:,i] = np.divide(exp_scores[:,i],(sum_exp_scores))
    #print(P[i])
    Py[i] = P[y[i],i]
	
    loss_array[i] = -1.0*np.log(Py[i])
    #print(loss_array[i])
	
    #P[:,i] = np.multiply(-1,np.log(exp_scores[y[i],i]/(sum_exp_scores)))
    #print(loss_array)
	
	   
  loss = np.divide(np.sum(loss_array), num_train)

  # Add regularization to the loss.
  loss = 0.5 * reg * np.sum(W * W) + loss

  
    # compute the gradient on scores
  dscores = P
  #print('P = ', P)
  #print('P shape = ', P.shape)
  for i in range(y.shape[0]):
       dscores[y[i],i]  = dscores[y[i],i] - 1.0
  #print('dscores = ', dscores)
  dscores /= X.shape[0]
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(np.transpose(X), np.transpose(dscores))
  #db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient
  
  
  
  
  
  
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
  
  
  #print('X shape = ', X.shape)  
  #print('X = ', X)
  #print(np.amax(X))
  #print(np.amin(X))
  #print('y shape = ', y.shape)
  #print('y = ', y)
  
  normalized_input = X
  
  pos_max = np.amax(X)
  neg_min = np.amin(X)
  
  for i in range (X.shape[0]): 
        for j in range (X.shape[1]): 
            if X[i,j] > 0: 
                normalized_input[i,j] = X[i,j] / pos_max
            elif X[i,j] < 0: 
                normalized_input[i,j] = X[i,j] / abs(neg_min)
  #print('normalized_input = ', normalized_input)
  
  #print(np.amax(normalized_input))
  #print(np.amin(normalized_input))
  
  
  scores = np.dot(np.transpose(W), np.transpose(normalized_input))
  scores = np.transpose(scores); 
  #print('scores shape = ', scores.shape)
  #print('scores = ', scores)

  
  softmax_input = scores
  #print('softmax_input shape is = ', softmax_input.shape)	
	
  exp_scores = np.exp(softmax_input)
  #print('exp of scores shape = ', exp_scores.shape )
  #print('exp of scores = ', exp_scores)
	
  sum_exp_scores = np.sum(exp_scores, axis = 1)
  #print('sum of exp scores = ', sum_exp_scores)
  #print('sum of exp scores shape = ', sum_exp_scores.shape)
	
	
  P = np.empty([softmax_input.shape[0], softmax_input.shape[1]])
  Py = np.empty([y.shape[0]])
  for i in range(y.shape[0]): 
       P[i,:] = np.divide(exp_scores[i,:],sum_exp_scores[i])
    
  #print("Probabilities = ", P)
  #print("P shape =", P.shape)
  for i in range(y.shape[0]): 
       Py[i] = P[i,y[i]]  
  #print('Py = ', Py)
  #print('Py shape = ', Py.shape)  
	
  data_loss = -1*np.log(Py)
  #print('data loss = ', data_loss)
  #print('data_loss shape = ', data_loss.shape)
	
  avg_data_loss =  (1/softmax_input.shape[0]) * np.sum(data_loss)
  #print('average data loss = ', avg_data_loss)
	
  reg_loss = (reg*np.sum(W**2))/2
  #print('regularization loss = ', reg_loss)
	
  loss = avg_data_loss + reg_loss
	
    # compute the gradient on scores
  dscores = P
  #print('P = ', P)
  #print('P shape = ', P.shape)
  for i in range(y.shape[0]):
       dscores[i,y[i]]  = dscores[i,y[i]] - 1.0
  #print('dscores = ', dscores)
  dscores /= X.shape[0]
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(np.transpose(X), dscores)
  #db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

