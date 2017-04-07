import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    print('CNN INIT')
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    pass
	
    W1 = np.random.normal(0,weight_scale,num_filters*filter_size*filter_size*3)
    W1 = np.reshape(W1, [num_filters,3,filter_size,filter_size])
    #print('W1 = ', W1)
    #print('W1 shape = ', W1.shape)
    b1 = np.zeros([num_filters])
    #print('b1 shape = ', b1.shape)
    W2 = np.random.normal(0,weight_scale,int(input_dim[1]/2)*int(input_dim[1]/2)*num_filters*hidden_dim)
    W2 = np.reshape(W2, [int(input_dim[1]/2)*int(input_dim[1]/2)*num_filters,hidden_dim])
   # print('W2 shape = ', W2.shape)
    b2 = np.zeros([hidden_dim])
    #print('b2 shape = ', b2.shape)
    W3 = np.random.normal(0,weight_scale,hidden_dim*num_classes)
    W3 = np.reshape(W3, [hidden_dim,num_classes])
    #print('W3 shape = ', W3.shape)
    b3 = np.zeros([num_classes])	
    #print('b3 shape = ', b3.shape)
	
    self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)		
     
	
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
	
    #print('W1 shape = ', W1.shape)
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    pass
	
	  #conv - relu - 2x2 max pool - affine - relu - affine - softmax

    # Layer1_out, _ = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    conv_out, _ = conv_forward_naive(X, W1, b1, conv_param)
    #print('conv_out shape = ', conv_out.shape)
    relu_out, _ = relu_forward(conv_out)
    Layer1_out, _ = max_pool_forward_naive(relu_out, pool_param)
    #print('Layer1_out =', Layer1_out[1,1,:,:])
    #print('Layer1_out shape = ', Layer1_out.shape)
	
    Layer2_out, _ = affine_relu_forward(Layer1_out, W2, b2)
    Layer3_out, _ = affine_forward(Layer2_out, W3, b3)
    scores = Layer3_out
    #print('scores shape = ', scores.shape)
    #print('scores = ', scores)
	  
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    pass
	
    #print("SOFTMAX ==================")
	#print('Y = ', y)
    #print('Y shape = ', y.shape)
	
    softmax_input = scores; 
    #print('softmax_input shape is = ', softmax_input.shape)	
	
    softmax_input_trans = np.transpose(softmax_input)
    #print('softmax_input_trans shape = ', softmax_input_trans.shape)
	
    exp_scores = np.exp(softmax_input_trans)
    #print('exp of scores = ', exp_scores)
	
    sum_exp_scores = np.sum(exp_scores, axis = 0)
    #print('sum of exp scores = ', sum_exp_scores)
	
	
    P = np.empty([softmax_input_trans.shape[0], softmax_input_trans.shape[1]])
    Py = np.empty([y.shape[0]])
    for i in range(y.shape[0]): 
         P[:,i] = exp_scores[:,i] / sum_exp_scores[i]
    
    #print("Probabilities = ", P)
    for i in range(y.shape[0]): 
         Py[i] = P[y[i],i]  
    #print('Py = ', Py)	
	
    data_loss = -1*np.log(Py)
    #print('data loss = ', data_loss)
    #print('reg = ', self.reg)
	
    avg_data_loss =  (1/softmax_input_trans.shape[1]) * np.sum(data_loss)
   # print('average data loss = ', avg_data_loss)
    #print('average data loss shape = ', avg_data_loss.shape)
	
    reg_loss = 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2)  + np.sum(W3**2))
    #print('regularization loss = ', reg_loss)
    #print('regularization loss = ', reg_loss.shape)
	
    loss = avg_data_loss + reg_loss
	
    #print('loss = ', loss)
	
    # compute the gradient on scores
    dscores = P
    #print('P = ', P)
    #print('P shape = ', P.shape)
    for i in range(y.shape[0]):
         dscores[y[i],i]  = dscores[y[i],i] - 1.0
    #print('dscores = ', dscores)
    dscores /= X.shape[0]
	
    _, cache = affine_forward(Layer2_out, W3, b3)
    dx3,dW3,db3 = affine_backward(np.transpose(dscores), cache)
		
    _ , cache = affine_relu_forward(Layer1_out, W2, b2)
    dx2,dW2,db2 = affine_relu_backward(dx3, cache)
		
    # _, cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    # dx1,dW1,db1 = conv_relu_pool_backward(dx2, cache)
    
    _, cache = max_pool_forward_naive(relu_out, pool_param)
    dx_max = max_pool_backward_naive(dx2, cache)
    
    _, cache = relu_forward(conv_out)
    dx_relu = relu_backward(dx_max, cache)
	
    _, cache = conv_forward_naive(X, W1, b1, conv_param)
    dx1,dW1,db1 = conv_backward_naive(dx_relu, cache)
		
    dW3 += self.reg*W3
    dW2 += self.reg*W2
    dW1 += self.reg*W1
	
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
	
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
