from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    numExamples = y.shape[0]
    numClasses = W.shape[1]

    mult = np.dot(X, W) # N x C

    for i in range(numExamples):

      fi = mult[i]
      fi -= np.max(fi)
      expsForiEx = np.exp(fi)
      loss += -fi[y[i]] + np.log(np.sum(expsForiEx))

      dW[:, y[i]] -= X[i]
      for j in range(numClasses):
        dW[:, j] += X[i] * expsForiEx[j] / np.sum(expsForiEx)
        
    dW /= numExamples
    dW += W * reg

    loss /= numExamples
    loss += np.sum(W * W) * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    numExamples = y.shape[0]
    numClasses = W.shape[1]

    mult = np.dot(X, W) # N x C
    
    mult -= np.max(mult, axis=1)[:, None]

    # from IPython.core.debugger import set_trace
    # set_trace()

    exps = np.exp(mult)
    sumExps = np.sum(exps, axis=1) # N x 1
    fOfTrue = mult[range(numExamples), y]
    losses = -fOfTrue + np.log(sumExps)
    loss = np.sum(losses)
    loss /= numExamples
    loss += np.sum(W * W) * reg

    expsDivSum = exps / sumExps[:, None] # Поделить заранее до матричного умножения
    dW = np.dot(X.T, expsDivSum)  # D x C

    mask = np.zeros(mult.shape)
    mask[range(numExamples), y] = 1
    
    # for i in range(numExamples):
    #   dW[:, y[i]] -= X[i]
    dW -= np.dot(X.T, mask)

    dW /= numExamples
    dW += W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
