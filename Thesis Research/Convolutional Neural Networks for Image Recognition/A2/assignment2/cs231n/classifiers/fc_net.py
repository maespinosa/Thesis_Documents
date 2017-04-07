import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
from cs231n.softmax import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    pass
	
    W1 = np.random.normal(0,weight_scale,input_dim*hidden_dim)
    W1 = np.reshape(W1,[input_dim,hidden_dim])
    b1 = np.zeros([hidden_dim])
    W2 = np.random.normal(0,weight_scale,hidden_dim*num_classes)
    W2 = np.reshape(W2,[hidden_dim,num_classes])
    b2 = np.zeros([num_classes])
	
    self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
	
    #print('W1 = ', W1)
    #print('W1 shape = ', W1.shape)
    #print('b1 = ', b1)
    #print('b1 shape = ', b1.shape)
    #print('W2 = ', W2)
    #print('W2 shape = ', W2.shape)
    #print('b2 = ', b2)
    #print('b2 shape = ', b2.shape)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    pass
    #print(np.amax(X))
    #print(np.amin(X))
    #print('x shape = ', X.shape)
    #print('W1 shape = ', self.params['W1'].shape)
    #print('b1 shape = ', self.params['b1'].shape)
    #print('W2 shape = ', self.params['W2'].shape)
    #print('b2 shape = ', self.params['b2'].shape)

    HL1_out, _ = affine_forward(X, self.params['W1'], self.params['b1'])
    #print('HL1_out = ', HL1_out)
    #print('HL1_out shape = ', HL1_out.shape)
	
    RELU1_out, _ = relu_forward(HL1_out)
	
    HL2_out, _ = affine_forward(RELU1_out, self.params['W2'], self.params['b2'])
	
    scores = HL2_out
	
    #print('scores = ', scores)
    #print('scores shape is = ', scores.shape)	
	


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores	
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
	

	
    #print('Y = ', y)
    #print('Y shape = ', y.shape)
	
    softmax_input = scores; 
    #print('softmax_input shape is = ', softmax_input.shape)	
	
    softmax_input_trans = np.transpose(softmax_input)
    #print('softmax_input_trans shape = ', softmax_input_trans.shape)
	
    #print('softmax input scores = ', softmax_input_trans)
	
    # max_input = np.amax(softmax_input_trans, axis = 0)
    # #print('max input shape = ', max_input.shape)
    # #print('max input = ', max_input)
	
    # max_vector = np.repeat(max_input, softmax_input_trans.shape[0], axis = 0)
    # max_matrix = np.reshape(max_vector ,(softmax_input_trans.shape[1], softmax_input_trans.shape[0]))
    # max_matrix = np.transpose(max_matrix)
    # #print('max_matrix shape =', max_matrix.shape)
    # #print('max_matrix = ', max_matrix)
	
    # normalized_input = softmax_input_trans - max_matrix #- softmax_input_trans
    # #print('normalized input = ', normalized_input)
	
    exp_scores = np.exp(softmax_input_trans)
    #print('exp of scores = ', exp_scores)
	
    sum_exp_scores = np.sum(exp_scores, axis = 0)
   # print('sum of exp scores = ', sum_exp_scores)
	
	
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
	
    reg_loss = (self.reg*np.sum(self.params['W2'] **2) + self.reg*np.sum(self.params['W1'] **2) )/2
    #print('regularization loss = ', reg_loss)
	
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
	
	
    _, cache = affine_forward(RELU1_out, self.params['W2'], self.params['b2'])
    dX_H2, dW2, db2 = affine_backward(np.transpose(dscores), cache)
	
    _, cache = relu_forward(HL1_out)
    dX_RELU = relu_backward(dX_H2, cache)

    _, cache = affine_forward(X, self.params['W1'], self.params['b1'])
    dX_H1, dW1, db1 = affine_backward(dX_RELU, cache)
	
    dW2 += self.reg*self.params['W2']
    dW1 += self.reg*self.params['W1']
	
	
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
				

class FullyConnectedNet(object):		
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
	
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float, seed=None):
    """
    Initialize a new FullyConnectedNet.

    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    pass


    if self.num_layers == 2: 
        
        W1 = np.random.normal(0,weight_scale,input_dim*hidden_dims[0])
        W1 = np.reshape(W1,[input_dim,hidden_dims[0]])
        b1 = np.zeros([hidden_dims[0]])
		
        W2 = np.random.normal(0,weight_scale,hidden_dims[0]*num_classes)
        W2 = np.reshape(W2,[hidden_dims[0],num_classes])
        b2 = np.zeros([num_classes])
		
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
		
    elif self.num_layers == 3: 

        W1 = np.random.normal(0,weight_scale,input_dim*hidden_dims[0])
        W1 = np.reshape(W1,[input_dim,hidden_dims[0]])
        b1 = np.zeros([hidden_dims[0]])
		
        W2 = np.random.normal(0,weight_scale,hidden_dims[0]*hidden_dims[1])
        W2 = np.reshape(W2,[hidden_dims[0],hidden_dims[1]])
        b2 = np.zeros([hidden_dims[1]])
		
        W3 = np.random.normal(0,weight_scale,hidden_dims[1]*num_classes)
        W3 = np.reshape(W3,[hidden_dims[1],num_classes])
        b3 = np.zeros([num_classes])
		
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
		
    elif self.num_layers == 4: 
	
        W1 = np.random.normal(0,weight_scale,input_dim*hidden_dims[0])
        W1 = np.reshape(W1,[input_dim,hidden_dims[0]])
        b1 = np.zeros([hidden_dims[0]])
		
        W2 = np.random.normal(0,weight_scale,hidden_dims[0]*hidden_dims[1])
        W2 = np.reshape(W2,[hidden_dims[0],hidden_dims[1]])
        b2 = np.zeros([hidden_dims[1]])
		
        W3 = np.random.normal(0,weight_scale,hidden_dims[1]*hidden_dims[2])
        W3 = np.reshape(W3,[hidden_dims[1],hidden_dims[2]])
        b3 = np.zeros([hidden_dims[2]])
		
        W4 = np.random.normal(0,weight_scale,hidden_dims[2]*num_classes)
        W4 = np.reshape(W4,[hidden_dims[2],num_classes])
        b4 = np.zeros([num_classes])
		
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4':W4, 'b4': b4}
		
    elif self.num_layers == 5: 
	
        W1 = np.random.normal(0,weight_scale,input_dim*hidden_dims[0])
        W1 = np.reshape(W1,[input_dim,hidden_dims[0]])
        b1 = np.zeros([hidden_dims[0]])
		
        W2 = np.random.normal(0,weight_scale,hidden_dims[0]*hidden_dims[1])
        W2 = np.reshape(W2,[hidden_dims[0],hidden_dims[1]])
        b2 = np.zeros([hidden_dims[1]])
		
        W3 = np.random.normal(0,weight_scale,hidden_dims[1]*hidden_dims[2])
        W3 = np.reshape(W3,[hidden_dims[1],hidden_dims[2]])
        b3 = np.zeros([hidden_dims[2]])
		
        W4 = np.random.normal(0,weight_scale,hidden_dims[2]*hidden_dims[3])
        W4 = np.reshape(W4,[hidden_dims[2],hidden_dims[3]])
        b4 = np.zeros([hidden_dims[3]])
		
        W5 = np.random.normal(0,weight_scale,hidden_dims[3]*num_classes)
        W5 = np.reshape(W5,[hidden_dims[3],num_classes])
        b5 = np.zeros([num_classes])
		
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4':W4, 'b4': b4, 'W5': W5, 'b5': b5}
		
    elif self.num_layers == 6: 
	
        W1 = np.random.normal(0,weight_scale,input_dim*hidden_dims[0])
        W1 = np.reshape(W1,[input_dim,hidden_dims[0]])
        b1 = np.zeros([hidden_dims[0]])
		
        W2 = np.random.normal(0,weight_scale,hidden_dims[0]*hidden_dims[1])
        W2 = np.reshape(W2,[hidden_dims[0],hidden_dims[1]])
        b2 = np.zeros([hidden_dims[1]])
		
        W3 = np.random.normal(0,weight_scale,hidden_dims[1]*hidden_dims[2])
        W3 = np.reshape(W3,[hidden_dims[1],hidden_dims[2]])
        b3 = np.zeros([hidden_dims[2]])
		
        W4 = np.random.normal(0,weight_scale,hidden_dims[2]*hidden_dims[3])
        W4 = np.reshape(W4,[hidden_dims[2],hidden_dims[3]])
        b4 = np.zeros([hidden_dims[3]])
		
        W5 = np.random.normal(0,weight_scale,hidden_dims[3]*hidden_dims[4])
        W5 = np.reshape(W5,[hidden_dims[3],hidden_dims[4]])
        b5 = np.zeros([hidden_dims[4]])
		
        W6 = np.random.normal(0,weight_scale,hidden_dims[4]*num_classes)
        W6 = np.reshape(W6,[hidden_dims[4],num_classes])
        b6 = np.zeros([num_classes])
		
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4':W4, 'b4': b4, 'W5': W5, 'b5': b5, 'W6': W6, 'b6': b6}

	
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    pass
	
	
    #print(np.amax(X))
    #print(np.amin(X))
    #print('x shape = ', X.shape)
    #print('W1 shape = ', self.params['W1'].shape)
    #print('b1 shape = ', self.params['b1'].shape)
    #print('W2 shape = ', self.params['W2'].shape)
    #print('b2 shape = ', self.params['b2'].shape)
	
    if self.num_layers == 2: 
        HL1_out, _ = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        HL2_out, _ = affine_forward(HL1_out, self.params['W2'], self.params['b2'])
        scores = HL2_out
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
    elif self.num_layers == 3: 
        HL1_out, _ = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        HL2_out, _ = affine_relu_forward(HL1_out,self.params['W2'], self.params['b2'])
        HL3_out, _ = affine_forward(HL2_out, self.params['W3'], self.params['b3'])
        scores = HL3_out
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        W3 = self.params['W3']
        b3 = self.params['b3']
    elif self.num_layers == 4: 
        HL1_out, _ = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        HL2_out, _ = affine_relu_forward(HL1_out,self.params['W2'], self.params['b2'])
        HL3_out, _ = affine_relu_forward(HL2_out,self.params['W3'], self.params['b3'])
        HL4_out, _ = affine_forward(HL3_out, self.params['W4'], self.params['b4'])
        scores = HL4_out
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        W3 = self.params['W3']
        b3 = self.params['b3']
        W4 = self.params['W4']
        b4 = self.params['b4']
    elif self.num_layers == 5: 
        HL1_out, _ = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        HL2_out, _ = affine_relu_forward(HL1_out,self.params['W2'], self.params['b2'])
        HL3_out, _ = affine_relu_forward(HL2_out,self.params['W3'], self.params['b3'])
        HL4_out, _ = affine_relu_forward(HL3_out,self.params['W4'], self.params['b4'])
        HL5_out, _ = affine_forward(HL4_out, self.params['W5'], self.params['b5'])
        scores = HL5_out
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        W3 = self.params['W3']
        b3 = self.params['b3']
        W4 = self.params['W4']
        b4 = self.params['b4']
        W5 = self.params['W5']
        b5 = self.params['b5']
    elif self.num_layers == 6: 
        HL1_out, _ = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        HL2_out, _ = affine_relu_forward(HL1_out,self.params['W2'], self.params['b2'])
        HL3_out, _ = affine_relu_forward(HL2_out,self.params['W3'], self.params['b3'])
        HL4_out, _ = affine_relu_forward(HL3_out,self.params['W4'], self.params['b4'])
        HL5_out, _ = affine_relu_forward(HL4_out,self.params['W5'], self.params['b5'])
        HL6_out, _ = affine_forward(HL5_out, self.params['W6'], self.params['b6'])
        scores = HL6_out
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        W3 = self.params['W3']
        b3 = self.params['b3']
        W4 = self.params['W4']
        b4 = self.params['b4']
        W5 = self.params['W5']
        b5 = self.params['b5']
        W6 = self.params['W6']
        b6 = self.params['b6']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
	
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
	
    if self.num_layers == 2: 
        reg_loss = 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2))
    elif self.num_layers == 3: 
        reg_loss = 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    elif self.num_layers == 4: 
        reg_loss = 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    elif self.num_layers == 5: 
        reg_loss = 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2) + np.sum(W5**2))
    elif self.num_layers == 6: 
        reg_loss = 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2) + np.sum(W5**2) + np.sum(W6**2))
		
    #reg_loss = (self.reg*np.sum(self.params['W2'] **2) + self.reg*np.sum(self.params['W1'] **2) )/2
    #print('regularization loss = ', reg_loss)
	
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
	
    if self.num_layers == 2: 
	
        _ , cache = affine_forward(HL1_out, self.params['W2'], self.params['b2'])
        dx2,dW2,db2 = affine_backward(np.transpose(dscores), cache)
		
        _, cache = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        dx1,dW1,db1 = affine_relu_backward(dx2, cache)
		
        dW2 += self.reg*self.params['W2']
        dW1 += self.reg*self.params['W1']
	
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    elif self.num_layers == 3: 
	
        _ , cache = affine_forward(HL2_out, self.params['W3'], self.params['b3'])
        dx3,dW3,db3 = affine_backward(np.transpose(dscores), cache)
		
        _ , cache = affine_relu_forward(HL1_out, self.params['W2'], self.params['b2'])
        dx2,dW2,db2 = affine_relu_backward(dx3, cache)
		
        _, cache = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        dx1,dW1,db1 = affine_relu_backward(dx2, cache)
		
        dW3 += self.reg*self.params['W3']
        dW2 += self.reg*self.params['W2']
        dW1 += self.reg*self.params['W1']
	
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
		
    elif self.num_layers == 4: 

        _ , cache = affine_forward(HL3_out, self.params['W4'], self.params['b4'])
        dx4,dW4,db4 = affine_backward(np.transpose(dscores), cache)
		
        _ , cache = affine_relu_forward(HL2_out, self.params['W3'], self.params['b3'])
        dx3,dW3,db3 = affine_relu_backward(dx4, cache)
		
        _ , cache = affine_relu_forward(HL1_out, self.params['W2'], self.params['b2'])
        dx2,dW2,db2 = affine_relu_backward(dx3, cache)
		
        _, cache = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        dx1,dW1,db1 = affine_relu_backward(dx2, cache)
        
        dW4 += self.reg*self.params['W4']
        dW3 += self.reg*self.params['W3']
        dW2 += self.reg*self.params['W2']
        dW1 += self.reg*self.params['W1']
	
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4}
		
		
    elif self.num_layers == 5: 

        _ , cache = affine_forward(HL4_out, self.params['W5'], self.params['b5'])
        dx5,dW5,db5 = affine_backward(np.transpose(dscores), cache)

        _ , cache = affine_relu_forward(HL3_out, self.params['W4'], self.params['b4'])
        dx4,dW4,db4 = affine_relu_backward(dx5, cache)
		
        _ , cache = affine_relu_forward(HL2_out, self.params['W3'], self.params['b3'])
        dx3,dW3,db3 = affine_relu_backward(dx4, cache)
		
        _ , cache = affine_relu_forward(HL1_out, self.params['W2'], self.params['b2'])
        dx2,dW2,db2 = affine_relu_backward(dx3, cache)
		
        _, cache = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        dx1,dW1,db1 = affine_relu_backward(dx2, cache)
		
        dW5 += self.reg*self.params['W5']
        dW4 += self.reg*self.params['W4']
        dW3 += self.reg*self.params['W3']
        dW2 += self.reg*self.params['W2']
        dW1 += self.reg*self.params['W1']
	
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5}
		
		
    elif self.num_layers == 6: 

        _ , cache = affine_forward(HL5_out, self.params['W6'], self.params['b6'])
        dx6,dW6,db6 = affine_backward(np.transpose(dscores), cache)

        _ , cache = affine_relu_forward(HL4_out, self.params['W5'], self.params['b5'])
        dx5,dW5,db5 = affine_relu_backward(dx6, cache)
		
        _ , cache = affine_relu_forward(HL3_out, self.params['W4'], self.params['b4'])
        dx4,dW4,db4 = affine_relu_backward(dx5, cache)
		
        _ , cache = affine_relu_forward(HL2_out, self.params['W3'], self.params['b3'])
        dx3,dW3,db3 = affine_relu_backward(dx4, cache)
		
        _ , cache = affine_relu_forward(HL1_out, self.params['W2'], self.params['b2'])
        dx2,dW2,db2 = affine_relu_backward(dx3, cache)
		
        _, cache = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        dx1,dW1,db1 = affine_relu_backward(dx2, cache)
		
        dW6 += self.reg*self.params['W6']
        dW5 += self.reg*self.params['W5']
        dW4 += self.reg*self.params['W4']
        dW3 += self.reg*self.params['W3']
        dW2 += self.reg*self.params['W2']
        dW1 += self.reg*self.params['W1']
	
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5, 'W6': dW6, 'b6': db6}

    # _, cache = affine_forward(RELU1_out, self.params['W2'], self.params['b2'])
    # dX_H2, dW2, db2 = affine_backward(np.transpose(dscores), cache)
	
    # _, cache = relu_forward(HL1_out)
    # dX_RELU = relu_backward(dX_H2, cache)

    # _, cache = affine_forward(X, self.params['W1'], self.params['b1'])
    # dX_H1, dW1, db1 = affine_backward(dX_RELU, cache)
	
    # dW2 += self.reg*self.params['W2']
    # dW1 += self.reg*self.params['W1']
	
	
    # grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
	
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
