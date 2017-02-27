import numpy as np
import sys
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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)

    scores -= np.max(scores) # used for numerical stability
    exp_scores = np.exp(scores)
    y_score = scores[y[i]]
    sum_exp_score = np.sum(exp_scores)


    loss += - y_score + np.log(sum_exp_score)
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += X[i] * (exp_scores[j] / sum_exp_score - 1)
      else:  
        dW[:, j] += X[i] * exp_scores[j] / sum_exp_score

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
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
  num_train = X.shape[0]

  scores = X.dot(W)
  scores -= scores.max(1).reshape(-1,1) # used for numerical stability
  exp_scores = np.exp(scores)
  y_exp_score = exp_scores[np.arange(num_train), y]
  sum_exp_score = np.sum(exp_scores, axis=1)

  loss += - np.sum(np.log(y_exp_score / sum_exp_score))
  # print("y_exp_score", y_exp_score[0], sum_exp_score[0])
  # print("loss1", loss)

  dW =  X.T.dot(exp_scores / sum_exp_score.reshape(-1,1))
  y_mat = np.zeros(scores.shape)
  y_mat[np.arange(num_train), y] = 1
  dW -= X.T.dot(y_mat) 

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

