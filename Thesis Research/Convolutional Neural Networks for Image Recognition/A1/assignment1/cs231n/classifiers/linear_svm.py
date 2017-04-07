import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #print('W dimensions = ',W.shape)
  #print('X dimensions =', X.shape)
  # compute the loss and the gradient
  num_classes = W.shape[1]
  #print('num_classes = ', num_classes)
  num_train = X.shape[0]
  #print('num_train = ', num_train)
  loss = 0.0
  class_count = 0
  for i in range(num_train):
    scores = X[i].dot(W)
    #print('scores dimensions = ', scores.shape)
    #print('scores = ', scores)
    #print('i =', i, 'y = ', y[i])
    correct_class_score = scores[y[i]]
    #print('correct score = ', correct_class_score)
    class_count = 0
    for j in range(num_classes):
      #print('j = ', j)
     
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      #print('margin = ', margin)
      if margin > 0:
        loss = loss + margin
        class_count = class_count +1
        dW[:,j] = dW[:,j] + X[i]
        #print('class count', class_count)

   

    dW[:,y[i]] = dW[:,y[i]]+np.multiply(-1,np.multiply(X[i],class_count))
  #print(dW.shape)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW = np.divide(dW,num_train)
  #print(dW)

  # Add regularization to the loss.
  loss = 0.5 * reg * np.sum(W * W) + loss

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.  
  
  
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #print('W shape = ', W.shape)
  #print('X shape =', X.shape)
  scores = X.dot(W)

  #print('scores dimension = ', scores.shape)
  #print('y shape', y.shape)
  #print(y)
  #print(scores[y])
  
  correct_scores = np.zeros(y.shape)
  correct_scores_matrix = np.zeros(scores.shape)
  #print('correct_scores shape = ', correct_scores.shape)
  #print('correct_scores_matrix shape = ', correct_scores_matrix.shape)
  for i in range(y.shape[0]):
    correct_scores[i] = scores[i,y[i]]
  #correct_scores_matrix[:,0] = correct_scores
  #print(correct_scores)
  #print(correct_scores_matrix)
  correct_scores_matrix = np.tile(correct_scores,(W.shape[1],1))
  correct_scores_matrix = np.transpose(correct_scores_matrix)
  #print('correct_scores_matrix shape = ', correct_scores_matrix.shape)
  #print('correct_scores_matrix = ', correct_scores_matrix)
 
  margin = scores - correct_scores_matrix + np.ones(scores.shape)
  #print(margin)
  #print('margin shape', margin.shape) #500x10
  #loss = np.zeros(y.shape)
  #print('loss shape', loss.shape)
  margin = np.multiply(margin, np.where(margin >= 0,1,0))
  loss = np.sum(margin,axis = 1) - np.ones(y.shape)
  #print(loss)
  #print('loss shape = ',loss.shape) 	
  loss = np.sum(loss, axis = 0)	
  loss /= X.shape[0]
  
    # Add regularization to the loss.
  loss = 0.5 * reg * np.sum(W * W) + loss
  

  pass
   
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  class_count = np.zeros(y.shape)
  class_count = np.where(margin > 0,1,0)
  class_count = np.sum(class_count, axis = 1) - np.ones(y.shape)
  class_count = np.multiply(class_count, np.where(class_count >= 0,1,0))
  #print('class_count shape = ', class_count.shape)
  #print('class count = ', class_count)
  #print('margin = ', margin)
  margin = np.where(margin > 0,margin,0)
  later_margin = np.where(margin == 1,1,0)
  #print('later_margin = ',later_margin)
  margin = np.where(margin != 1,1,0)
 # print('margin = ', margin)
  dW = np.transpose(X).dot(margin)
  class_count = np.transpose(np.tile(class_count, (X.shape[1],1)))
  #print('class_count shape = ', class_count.shape)
  #print('class count = ', class_count)
  #print('X shape = ', X.shape)
  
  # test = np.tile(X,(10,1,1))
  # test1 = np.transpose(np.tile(margin,(3073,1,1)))
  # print(test.shape)
  # print(test1.shape)
  # test2 = np.multiply(test1,test)
  # print(test2.shape)
  # test3 = np.sum(test2,axis =1)
  # print(test3.shape)
  # dW = np.transpose(test3)
  # print(X)
  # print(class_count)
  # print(np.multiply(-1,np.multiply(X,class_count)))
  
  dW = np.add(dW,np.transpose(np.multiply(-1,np.multiply(X,class_count))).dot(later_margin))
  dW = np.divide(dW,X.shape[0]) 
  #print(np.transpose(np.multiply(-1,np.multiply(X,class_count))).dot(later_margin))
  
  #print(dW)
 # print('dW shape = ', dW.shape)
  
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
