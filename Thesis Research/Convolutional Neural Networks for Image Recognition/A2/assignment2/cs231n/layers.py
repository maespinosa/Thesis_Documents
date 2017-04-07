import numpy as np

from cs231n.fast_layers import *

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = np.empty([x.shape[0],b.shape[0]])
  
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  pass
  

  #print('x shape is: ', x.shape)
  #print('w shape is: ', w.shape)
  #print('b shape is : ', b.shape)
 
  x_reshaped = np.reshape(x,(x.shape[0],np.prod(x.shape[1:])))
  #print('x_reshaped shape is : ', x_reshaped.shape)
  #print('x_reshaped = ', x_reshaped)
  
  #b_tile = np.tile(np.transpose(b),(120,1))	
  #print(b_tile.shape)
  
  w_transposed = np.transpose(w)
  #print('w_transposed shape is : ', w_transposed.shape)
  
  xr_transposed = np.transpose(x_reshaped)
  #print('x_reshaped transposed shape is :', xr_transposed.shape)
  
  for i in range(x_reshaped.shape[0]):
			#print(np.dot(w_transposed,xr_transposed[:,i]).shape)
			out[i,:] = np.add(np.dot(w_transposed,xr_transposed[:,i]),b)
			#print('output shape is : ', out.shape) 

  #print('output is : ', out)

 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  pass
  #print('dout shape is: ', dout.shape)
  #print('b shape is: ', b.shape)
  
  x_reshaped = np.reshape(x,(x.shape[0],np.prod(x.shape[1:])))
  #print('x_reshaped shape is : ', x_reshaped.shape)
  xr_transposed = np.transpose(x_reshaped)
  #print('x_reshaped transposed shape is :', xr_transposed.shape)
  w_transposed = np.transpose(w)
  #print('w_transposed shape is : ', w_transposed.shape)

  dw = np.dot(xr_transposed, dout)
  #print('dw shape is : ', dw.shape)
  #6x5 = 6x10 * 10x5
  dx = np.dot(dout,w_transposed)
  dx = np.reshape(dx,x.shape)
  #print('dx shape is : ', dx.shape)
  #10x6 = 10x5 * 5x6 
  
  db = np.sum(dout, axis = 0)
  #print('db shape is : ', db.shape)
  #print('db is: ', db)
  #print('b is:',b)
  #5x1 = 
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  pass
 # print('x input is: ', x)
  #print('x shape: ', x.shape)
  out = x
  out[out <= 0] = 0
  #print('relu out is: ' , out)
 # print('relu out shape: ', out.shape)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  pass
  #print('dout shape = ', dout.shape)
  #print('dout = ', dout)
  
  #print('x shape is: ', x.shape)
  #print('x is: ', x)
  if len(x) > 1: 
      x_reshaped = np.reshape(x,(x.shape[0],np.prod(x.shape[1:])))
  
      dx = np.empty([x_reshaped.shape[0], x_reshaped.shape[1]])
      #print('x mask', (x>0))
      mask = np.zeros([x_reshaped.shape[0], x_reshaped.shape[1]])
      for i in range(x_reshaped.shape[0]):
           for j in range(x_reshaped.shape[1]):
                #print(x[i,j])
                if (x_reshaped[i,j] > 0.0): 
                    mask[i,j] =1
            # else: 
                # mask[i,j] = 1.0
      mask = np.reshape(mask, x.shape)
  else: 
  
      dx = np.empty([x.shape[0], x.shape[1]])
      #print('x mask', (x>0))
      mask = np.zeros([x.shape[0], x.shape[1]])
      for i in range(x.shape[0]):
           for j in range(x.shape[1]):
                #print(x[i,j])
                if (x[i,j] > 0.0): 
                    mask[i,j] =1
            # else: 
                # mask[i,j] = 1.0
     
  
  #print('mask = ', mask)
  #print('x = ', x)  
  #dx = dout
  #dout[x < 0] = 0
  
  #dx = np.multiply((x >= 0),dout)
  dx = np.multiply(mask,dout)
  #print('dx shape = ', dx.shape)
  #print('dx = ', dx)
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pass
  
  # N 		: Number of images
  # C 		: Channels 
  # H 		: Height 
  # W		: Width 
  # F 		: Number of filters
  # HH 		: Filter Height 
  # WW		: Filter Width 
  # pad		: Number of pixels to zero-pad input 
  # stride	: Number of pixels between adjacent receptive fields 
  #print('x = ', x)
  #print('x shape = ', x.shape)
  #print('b = ', b)
  
  #DISPLAY THE CRITICAL DIMENSIONS 
  pad = int(conv_param['pad'])
  #print('pad = ', pad)
  
  stride = int(conv_param['stride'])
  #print('stride = ', stride)
  
  # Input Volume Dimensions
  N = x.shape[0]
  #print('N = ', N)
  
  C = x.shape[1]
  #print('C = ', C)
  
  H = x.shape[2]
  #print('H = ', H)
  
  W = x.shape[3]
  #print('W = ', W)
  
  #Filter Dimensions
  F = w.shape[0]
  #print('F = ', F)
  
  HH = w.shape[2]
  #print('HH = ', HH)
  
  WW = w.shape[3]
  #print('WW = ', WW)
  
  #Output Volume Dimensions
  OH = 1 + ((H + 2 * pad - HH)/stride)
  #print('OH = ', OH)
  
  OW = 1 + ((W + 2 * pad - WW)/stride)
  #print('OW = ', OW) 
  
  #TAKE BLOCKS OF INPUT VOLUME AND RESHAPE 
  X_col = np.zeros([int(OH*OW),C*HH*WW])
  #print('X_col shape  = ', X_col.shape)
  
  w_row = np.zeros([F,HH*WW*C])	

  x_pad = np.zeros([1,(int(H)+(pad*2))*(int(W)+(pad*2))*int(C)])
  x_pad = np.reshape(x_pad, [C,(int(H)+(pad*2)), (int(W)+(pad*2))])


  #print('x_pad = ', x_pad)
  #print('x_pad shape = ', x_pad.shape)
  
  out = np.empty([N,F,int(OH),int(OW)]) * 0
  
  filter_w = np.empty([HH, WW]) * 0 
  #print('w = ', w)
  for ii in range(F): 
	    for iii in range(C): 
	        filter_w = w[ii,iii,:,:]
	        #print('filter_w = ', filter_w)
	        #print('filter_w shape = ', filter_w.shape)
	        filter_w = np.reshape(filter_w, [1,HH*WW])
	        #print('filter_w = ', filter_w)
	        w_row[ii,(iii*HH*WW):(iii*HH*WW)+HH*WW] = filter_w

	
  #print('w_row = ', w_row)
  #print('w_row shape = ', w_row.shape)
  
  for i in range(N): 
    #print('i = ', i)
    x_pad[:,pad:x_pad.shape[1]-pad,pad:x_pad.shape[2]-pad] = x[i,:,:,:]
    padded_x = x_pad
	 
    #print('padded_x shape = ', padded_x.shape)
    #print('padded_x = ', padded_x)
	
    loc_counter = 0

    j = 0
   # print('j = ', j)
    k = 0
    #print('k = ', k)
    horz_count = 0
    vert_count = 0
    while vert_count < int(OH):
	    
	    while horz_count < int(OW): 
		    
	        X_block = padded_x[:,j:j+HH,k:k+WW]
		    #print('X_block shape = ', X_block.shape)
	        #print('X_block= ', X_block)
	        X_block_col = np.reshape(X_block,(1,HH*WW*C))	
		    #print('X_block_col shape = ', X_block_col.shape)
		    #print('X_block_col = ', X_block_col)
	        X_col[loc_counter,:] = X_block_col
            #print('X_col = ', X_col)
	        k = k + stride
	        #print('k = ', k)
		    #print('loc_counter = ', loc_counter)
	        loc_counter = loc_counter + 1
	        horz_count = horz_count + 1
	        #print('horz_count = ', horz_count)
		    
	    k = 0
	    horz_count = 0
	    #print('k = ', k) 
	    j = j + stride 
	    #print('j = ', j)
	    vert_count = vert_count + 1
	    #print('vert_count = ', vert_count)
	    
				
    #print('X_col = ', X_col)
    #print('X_col shape = ', X_col.shape)
	
    conv_out = np.dot(w_row, np.transpose(X_col))
    #print('conv_out = ', conv_out)
    #print('conv_out shape = ', conv_out.shape)

    conv_out = np.reshape(conv_out, [F,int(OH),int(OW)])
    #print('conv_out = ', conv_out)
    #print('conv_out shape = ', conv_out.shape)
    iiii = 0
    for iiii in range(F):
         conv_out[iiii,:,:] = conv_out[iiii,:,:] + b[iiii]    
    #print('conv_out = ', conv_out)
    #print('conv_out shape = ', conv_out.shape)

  # x_reshaped = np.reshape(x,(x.shape[0],np.prod(x.shape[1:])))
  
    
    out[i,:,:,:] = conv_out
    #print('out shape = ', out.shape)
    #print('out = ', out)	
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  pass
  
  x, w, b, conv_param = cache
  
  #- out: Output data, of shape (N, F, H', W') where H' and W' are given by
  #  H' = 1 + (H + 2 * pad - HH) / stride
  #  W' = 1 + (W + 2 * pad - WW) / stride

  
  #DISPLAY THE CRITICAL DIMENSIONS 
   #=============================
  pad = int(conv_param['pad'])
  #print('pad = ', pad)
  
  stride = int(conv_param['stride'])
  #print('stride = ', stride)
  
# Input Volume Dimensions
  N = dout.shape[0]
  #print('N = ', N)
  
  H = dout.shape[2]
  #print('H = ', H)
  
  W = dout.shape[3]
  #print('W = ', W)
  
  #Filter Dimensions
  F = w.shape[0]
  #print('F = ', F)
  
  C = w.shape[1]
  #print('C = ', C)
  
  HH = w.shape[2]
  #print('HH = ', HH)
  
  WW = w.shape[3]
  #print('WW = ', WW)
  
  #Output Volume Dimensions
  OH = 1 + ((H + 2 * pad - HH)/stride)
  #print('OH = ', OH)
  
  OW = 1 + ((W + 2 * pad - WW)/stride)
  #print('OW = ', OW) 
  
  # FIND DX 
  #=============================
  #=============================
  #print('X_col shape  = ', X_col.shape)
  w_row = np.zeros([F,HH*WW*C])	
  #w_row = w_row * 0
  
  out = np.zeros([N,C,int(OH),int(OW)])
  
  #ROTATE THE WEIGHTS
  #=======================
  #print('w = ', w)
  
  w_rot180 = w * 0
  #print('w_rot180 shape = ', w_rot180.shape)			
  
  for i in range(F): 
       for j in range(C): 
            w_rot180[i,j,:,:] = np.rot90(w[i,j,:,:], 2)
				
  #print('w_rot180 = ', w_rot180)
  
  #RACK AND STACK WEIGHTS INTO ROW VECTORS 
  #=============================
  filter_w = np.empty([HH, WW]) * 0 
  
  for ii in range(F): 
	    for iii in range(C): 
	        filter_w = w_rot180[ii,iii,:,:]
	        #print('filter_w = ', filter_w)
	        #print('filter_w shape = ', filter_w.shape)
	        filter_w = np.reshape(filter_w, [1,HH*WW])
	        #print('filter_w = ', filter_w)
	        w_row[ii,(iii*HH*WW):(iii*HH*WW)+HH*WW] = filter_w

	
  #print('w_row = ', w_row)
  #print('w_row shape = ', w_row.shape)
  
  
  
    #INITIALIZE COLUMN SIZE
  X_col = np.zeros([int(OH*OW),HH*WW])
  #X_col = X_col * 0
  
  #INITIALIZE PADDED MATRIX 
  x_pad = np.zeros([1,(int(H)+(pad*2))*(int(W)+(pad*2))])
  x_pad = np.reshape(x_pad, [(H+(pad*2)), (W+(pad*2))])
  #x_pad = x_pad * 0 

  #print('x_pad = ', x_pad)
  #print('x_pad shape = ', x_pad.shape)
  
  
  conv = np.zeros([F,C,int(OH),int(OW)])
  conv_out = np.zeros([C,int(OH),int(OW)])
  
  #CONVOLVE
  #=============================
  for i in range(N):  #NUMBER OF SAMPLES 
       for ff in range (F):  #NUMBER OF FILTERS
	        
			#PLACE X DATA INTO PADDED MATRIX 
            x_pad[pad:x_pad.shape[0]-pad,pad:x_pad.shape[1]-pad] = dout[i,ff,:,:]
            padded_x = x_pad
            #print('x_pad = ', x_pad)
			
	        # INITIALIZE COUNTERS
			
            loc_counter = 0
            j = 0
            k = 0
            horz_count = 0
            vert_count = 0
			
			#RACK AND STACK INPUT DATA INTO COLUMNS
            while vert_count < int(OH):
	    
	             while horz_count < int(OW): 
		    
	                X_block = padded_x[j:j+HH,k:k+WW]
		            #print('X_block shape = ', X_block.shape)
	                #print('X_block= ', X_block)
	                X_block_col = np.reshape(X_block,(1,HH*WW))	
		            #print('X_block_col shape = ', X_block_col.shape)
		            #print('X_block_col = ', X_block_col)
	                X_col[loc_counter,:] = X_block_col
			
	                k = k + stride
	                #print('k = ', k)
		            #print('loc_counter = ', loc_counter)
	                loc_counter = loc_counter + 1
	                horz_count = horz_count + 1
	                #print('horz_count = ', horz_count)
		    
	             k = 0
	             horz_count = 0
	             #print('k = ', k) 
	             j = j + stride 
	             #print('j = ', j)
	             vert_count = vert_count + 1
	             #print('vert_count = ', vert_count)
	    
				
            #print('X_col = ', X_col)
            #print('X_col shape = ', X_col.shape)
	        
			#DOT PRODUCT OF FILTER AND X DATA 
            dot1 = np.dot(w_row[ff,0:(HH*WW*1)], np.transpose(X_col))
            dot1 = np.reshape(dot1, [int(OH),int(OW)])
            #print('dot1 = ', dot1)
            #print('dot1 shape = ', dot1.shape)
            
            dot2 = np.dot(w_row[ff,HH*WW*1:(HH*WW*2)], np.transpose(X_col))
            dot2 = np.reshape(dot2, [int(OH),int(OW)])
            #print('dot2 = ', dot2)
            #print('dot2 shape = ', dot2.shape)
			
            dot3 = np.dot(w_row[ff,HH*WW*2:(HH*WW*3)], np.transpose(X_col))
            dot3 = np.reshape(dot3, [int(OH),int(OW)])
            #print('dot3  = ', dot3)
			
            conv[ff,0,:,:] = dot1
            conv[ff,1,:,:] = dot2
            conv[ff,2,:,:] = dot3
	   
       conv_out = np.sum(conv, axis = 0)	   
       #print('conv_out = ', conv_out)
       #print('conv_out shape = ', conv_out.shape)
       out[i,:,:,:] = conv_out
	   #print('out shape = ', out.shape)
	   #print('out = ', out)	
  #print('out = ', out)
  dx = out
  #print('dx = ', dx)
  
 #FIND DW 
  #=============================
  #=============================
  
  # DOUT IS NEW FILTER  
  
  # Input Volume Dimensions
  N = x.shape[0]
  #print('N = ', N)
  
  C = x.shape[1]
  #print('C = ', C)
  
  H = x.shape[2]
  #print('H = ', H)
  
  W = x.shape[3]
  #print('W = ', W)
  
  
  #Filter Dimensions
  F = dout.shape[1]
  #print('F = ', F)
  
  HH = dout.shape[2]
  #print('HH = ', HH)
  
  WW = dout.shape[3]
  #print('WW = ', WW)
  
  #Output Volume Dimensions
  OH = 1 + ((H + 2 * pad - HH)/stride)
  #print('OH = ', OH)
  
  OW = 1 + ((W + 2 * pad - WW)/stride)
  #print('OW = ', OW) 
  
  #ROTATE THE WEIGHTS
  #=======================
  #print('dout = ', dout)
  
  dout_rot180 = dout * 0
  #print('dout_rot180 shape = ', dout_rot180.shape)			
  
  for i in range(N): 
       for j in range(F): 
            dout_rot180[i,j,:,:] = np.rot90(dout[i,j,:,:], 2)
				
  #print('dout_rot180 = ', dout_rot180)  
  #RACK AND STACK WEIGHTS INTO ROW VECTORS 
  #=============================
  filter_w = np.zeros([HH, WW])
  
  w_row = np.zeros([N,F,HH*WW])	
 
  for i in range (N): 
       for ii in range(F): 
	         
	             filter_w = dout[i,ii,:,:]
	             #print('filter_w = ', filter_w)
	             #print('filter_w shape = ', filter_w.shape)
	             filter_w = np.reshape(filter_w, [1,HH*WW])
	             #print('filter_w = ', filter_w)
	             #w_row[i,ii,(iii*HH*WW):(iii*HH*WW)+HH*WW] = filter_w
	             w_row[i,ii,:] = filter_w
	
  #print('w_row = ', w_row)
  #print('w_row shape = ', w_row.shape)
  
  
  
    #INITIALIZE COLUMN SIZE
  X_col0 = np.zeros([int(OH*OW),HH*WW])
  X_col1 = np.zeros([int(OH*OW),HH*WW])
  X_col2 = np.zeros([int(OH*OW),HH*WW])
  
  #INITIALIZE PADDED MATRIX 
  x_pad = np.zeros([1,(int(H)+(pad*2))*(int(W)+(pad*2))*C])
  x_pad= np.reshape(x_pad, [C,(int(H)+(int(pad)*2)), (int(W)+(int(pad)*2))])

  
  #print('x_pad = ', x_pad)
  #print('x_pad shape = ', x_pad.shape)
  
  
  conv = np.zeros([N,F,C,int(OH),int(OW)])
  conv_out = np.zeros([F,C,int(OH),int(OW)])
  
  #CONVOLVE
  #=============================
  for i in range(N):  #NUMBER OF SAMPLES 
       for ff in range (F):  #NUMBER OF FILTERS
	        
			#PLACE X DATA INTO PADDED MATRIX 
            x_pad[:,int(pad):x_pad.shape[1]-int(pad),int(pad):x_pad.shape[2]-int(pad)] = x[i,:,:,:]
            padded_x = x_pad
			
	        # INITIALIZE COUNTERS
			
            loc_counter = 0
            j = 0
            k = 0
            horz_count = 0
            vert_count = 0
			
			#RACK AND STACK INPUT DATA INTO COLUMNS
            while vert_count < int(OH):
	    
	             while horz_count < int(OW): 
		    
	                 X_block0 = padded_x[0,j:j+HH,k:k+WW]
	                 X_block1 = padded_x[1,j:j+HH,k:k+WW]
	                 X_block2 = padded_x[2,j:j+HH,k:k+WW]
					
		            #print('X_block shape = ', X_block.shape)
	                #print('X_block= ', X_block)
	                 X_block0_col = np.reshape(X_block0,(1,HH*WW))	
	                 X_block1_col = np.reshape(X_block1,(1,HH*WW))	
	                 X_block2_col = np.reshape(X_block2,(1,HH*WW))	
					
		            #print('X_block_col shape = ', X_block_col.shape)
		            #print('X_block_col = ', X_block_col)
	                 X_col0[loc_counter,:] = X_block0_col
	                 X_col1[loc_counter,:] = X_block1_col
	                 X_col2[loc_counter,:] = X_block2_col
					
	                 k = k + stride
	                #print('k = ', k)
		            #print('loc_counter = ', loc_counter)
	                 loc_counter = loc_counter + 1
	                 horz_count = horz_count + 1
	                #print('horz_count = ', horz_count)
		    
	             k = 0
	             horz_count = 0
	             #print('k = ', k) 
	             j = j + stride 
	             #print('j = ', j)
	             vert_count = vert_count + 1
	             #print('vert_count = ', vert_count)
	    
				
            #print('X_col = ', X_col)
            #print('X_col shape = ', X_col.shape)	
            #print('w_row shape = ', w_row.shape)
			
			#DOT PRODUCT OF FILTER AND X DATA 
            #dot0 = np.dot(w_row[i,ff,:], np.transpose(X_col0))
            dot0 = np.dot(X_col0, w_row[i,ff,:])
            dot0 = np.reshape(dot0, [int(OH),int(OW)])
            #print('dot0 shape = ', dot0.shape)
		
            #dot1 = np.dot(w_row[i,ff,:], np.transpose(X_col1))
            dot1 = np.dot(X_col1, w_row[i,ff,:])
            dot1 = np.reshape(dot1, [int(OH),int(OW)])
            #print('dot1 shape = ', dot1.shape)
			
            #dot2 = np.dot(w_row[i,ff,:], np.transpose(X_col2))
            dot2 = np.dot(X_col2, w_row[i,ff,:])
            dot2 = np.reshape(dot2, [int(OH),int(OW)])
            #print('dot2 shape = ', dot2.shape)
			
		
            conv[i,ff,0,:,:] = dot0
            conv[i,ff,1,:,:] = dot1
            conv[i,ff,2,:,:] = dot2

  conv_out = np.sum(conv, axis = 0)
  
  dw = conv_out
  #print('dw = ', dw)
  #print('dw shape = ', dw.shape)
  
  
  
 #FIND DB
  #=============================
  #=============================
  db = np.zeros([N,F,HH,WW])
  for i in range(N):
      for j in range(F): 
          db[i,j,:,:] = 1 * dout[i,j,:,:]
	   
  #print('db shape = ', db.shape)	
  
  db = np.sum(db,axis = 0)
  #print('db shape = ', db.shape)	
  db = np.sum(db,axis = 2)
  #print('db shape = ', db.shape)	
  db = np.sum(db,axis = 1)
  #print('db shape = ', db.shape)	
  
  
 
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  
  #INPUT VALUES AND DIMENSIONS
  #print('x = ', x)
  #print('x shape = ', x.shape)
  
  N = x.shape[0]
  #print('N = ', N)
  
  C = x.shape[1]
  #print('C = ', C)
  
  H = x.shape[2]
  #print('H = ', H)
  
  W = x.shape[3]
  #print('W = ', W)

  PW = pool_param['pool_width']
  PH = pool_param['pool_height']
  stride = pool_param['stride']
  
  x_loc = int(((W-2)/stride) +1)
  #print('PW = ', PW)
  y_loc = int(((H-2)/stride) +1)
  #print('PH = ', PH)
  
  #print('stride =', stride)
  
  D = C
  #print('pool depth = ', D)
  
  #CALCULATIONS 
  
  max_pool = np.zeros([D,y_loc, x_loc])
  #print('max_pool shape = ', max_pool.shape)
  
  max_all = np.zeros([N,np.prod(max_pool.shape)])
  #print('max_all = ', max_all.shape)
  
  y_index = 0 
  x_index = 0 
  pool_y_loc = 0 
  pool_x_loc = 0
  
  for i in range(N): # Number of images
    for j in range(C): # RGB colors 
        while pool_y_loc < y_loc:
            while pool_x_loc < x_loc:
                max_pool[j,pool_y_loc,pool_x_loc] = np.amax(x[i,j, y_index:y_index+PH,x_index:x_index + PW])
                #print('max_pool = ', max_pool)
				
				
				
                x_index = x_index + stride
                #print('x_index = ', x_index)
				
                pool_x_loc = pool_x_loc + 1
               # print('pool_x_loc = ', pool_x_loc)
				
            x_index = 0
            pool_x_loc = 0
			
            y_index = y_index + stride 
            pool_y_loc = pool_y_loc + 1
            #print('pool_y_loc = ', pool_y_loc)			
		
        y_index = 0
        x_index = 0
        pool_y_loc = 0
        pool_x_loc = 0
        max_reshape = np.reshape(max_pool, [1,np.prod(max_pool.shape)])
        #print('max_reshape shape = ', max_reshape.shape)
    max_all[i,:] = max_reshape
  out = np.reshape(max_all, [N,C,y_loc,x_loc])
  #out = max_all
  #print('out shape= ', out.shape)
  #print('out = ', out)
		
				
		
	
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  pass
  
  #print('dout shape = ', dout.shape)
  #print('dout = ', dout)
  
  x, pool_param = cache 
  
  dx = x*0
  
  N = x.shape[0]
  #print('N = ', N)
  
  C = x.shape[1]
 # print('C = ', C)
  
  H = x.shape[2]
  #print('H = ', H)
  
  W = x.shape[3]
  #print('W = ', W)
  
  F = 2
  
  PW = pool_param['pool_width']
  PH = pool_param['pool_height']
  stride = pool_param['stride']
  
  x_loc = int(((W-F)/stride) +1)
 # print('x_loc = ', x_loc)
  y_loc = int(((H-F)/stride) +1)
  #print('y_loc = ', y_loc)
  
  #print('stride =', stride)
  
  out , _ = max_pool_forward_naive(x, pool_param)
  #print('out shape = ', out.shape)
  y_index = 0 
  x_index = 0 
  pool_y_loc = 0 
  pool_x_loc = 0
  
  for i in range(N): # Number of images
    for j in range(C): # RGB colors 
        while pool_y_loc < y_loc:
            while pool_x_loc < x_loc:
                pool_block = x[i,j, y_index:y_index+PH,x_index:x_index + PW]
                #print('pool_block = ', pool_block)
                pool_block[pool_block == out[i,j,pool_y_loc,pool_x_loc]] = 1
                pool_block[pool_block != 1] = 0
                pool_block[pool_block == 1] = dout[i,j,pool_y_loc,pool_x_loc]
                #print('out = ', out[i,j,pool_y_loc,pool_x_loc])
                #print('pool_block = ', pool_block)
				
                dx[i,j, y_index:y_index+PH,x_index:x_index + PW] = pool_block
				
					
                x_index = x_index + stride
               # print('x_index = ', x_index)
				
                pool_x_loc = pool_x_loc + 1
                #print('pool_x_loc = ', pool_x_loc)
				
            x_index = 0
            pool_x_loc = 0
			
            y_index = y_index + stride 
            pool_y_loc = pool_y_loc + 1
            #print('pool_y_loc = ', pool_y_loc)			
		
        y_index = 0
        x_index = 0
        pool_y_loc = 0
        pool_x_loc = 0

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  #print('correct_class_scores shape = ', correct_class_scores.shape)
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  #print('margins shape = ', margins.shape)
  margins[np.arange(N), y] = 0
  #print('margins shape = ', margins.shape)
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
