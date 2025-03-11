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
  
  num_dimensions = W.shape[0]
  num_classes = W.shape[1]
  num_train = X.shape[0]


  for i in range(num_train):
      
      scores = X[i].dot(W)
      
      # to avoid the numerical instability issues
      scores -= np.max(scores)

      # exponent sum over all classes
      exponent_sum = np.sum(np.exp(scores))

      softmax_probability = lambda k: np.exp(scores[k])/exponent_sum

      loss += -np.log(softmax_probability(y[i])) # -ve log likelihood of correct class
      
     
      for j in range(num_classes):
          
          if(j == y[i]):
              
              dW[:, j] += X[i]*(softmax_probability(j) - 1)
              
          else:
              
              dW[:, j] += X[i]*(softmax_probability(j))

    
  ## normalize the loss over all training samples
  
  
  loss /= num_train
  dW /= num_train
  
  loss += reg*np.sum(W*W)
  dW += 2*reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  
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

  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  scores = X.dot(W)
#   print(scores.shape)
  scores -= np.max(scores) 
  
#   print(scores.shape)
  exponent_sum = np.sum(np.exp(scores), axis = 1, keepdims = True)
  
  softmax_probability = np.exp(scores)/exponent_sum
  
#   print(softmax_probability.shape)
  
  loss = np.sum(-np.log(softmax_probability[np.arange(num_train), y])) 
  # we want N x N shape for softmax_probability to use arange function to avoid explicit for loop over the num_samples
  
  
  
  ## gradient calculation
  ## gradient will equal the softmax_probability where the label is not equal to y
  ## else a 1 will be subtracted from the softmax_probability value
  
  temp = np.zeros_like(softmax_probability)
  
  temp[np.arange(num_train), y] = 1
  
  dW = X.T.dot(softmax_probability - temp) ## D x C dimensions to match the dimensions of W
  
  
  ##normalize loss
  loss /= num_train
  dW /= num_train
  
  loss += reg*(np.sum(W*W))
  dW += 2*reg*W
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

