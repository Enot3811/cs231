from builtins import range
import numpy as np



def affine_forward(X, W, bias):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - X: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - W: A numpy array of weights, of shape (D, M)
    - bias: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (X, W, bias)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    X_rows = np.copy(X).reshape(N, -1) # N x D
    X_rows = np.hstack((X_rows, np.ones((N, 1)))) # N x D+1
    W_bias = np.vstack((W, bias.T)) # D+1 x M
    z = np.dot(X_rows, W_bias)
    out = z

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (X, W, bias)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - X: Input data, of shape (N, d_1, ... d_k)
      - W: Weights, of shape (D, M)
      - bias: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to X, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to W, of shape (D, M)
    - db: Gradient with respect to bias, of shape (M,)
    """
    X, W, bias = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    M = bias.shape[0]

    X_rows = np.copy(X).reshape(N, -1)
    X_rows = np.hstack((X_rows, np.ones((N, 1)))) # N x D+1
    W_bias = np.vstack((W, bias.T)) # D+1 x M

    dWb = np.dot(X_rows.T, dout) # D+1 x M
    dw = dWb[0:-1, :] # D x M
    db = dWb[-1, :].reshape(M,) # M x ,

    dX_rows = np.dot(dout, W_bias.T) # N x D+1
    dx = dX_rows[:, 0:-1] # N x D
    dx = dx.reshape(X.shape)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.copy(x)
    out[out < 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    dx, X = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # from IPython.core.debugger import set_trace
    # set_trace()

    dx = np.copy(dout)
    mask = np.float64(X > 0)
    #mask = np.float64(x > 0)
    #mask[mask==0] = 0.5
    dx *= mask

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(X, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement loss and gradient for multiclass SVM classification.    #
    # This will be similar to the svm loss vectorized implementation in       #
    # cs231n/classifiers/linear_svm.py.                                       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = X.shape[1]
    h = 1

    correct_class = X[range(N), y] # N x ,
    margins = X - correct_class[:, None] + h # N x C
    margins[margins < 0] = 0
    margins[range(N), y] = 0
    loss = np.sum(margins) / N

    dx = np.zeros_like(X)
        
    mask = np.float64(margins > 0) # N x C
    mask[range(N), y] = -1.0 * np.sum(mask, axis=1)
    dx = mask / N


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(X, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = X.shape[1]

    # from IPython.core.debugger import set_trace
    # set_trace()
    sub = np.copy(X) - np.max(X, axis=1)[:, None]

    exps = np.exp(sub)
    true_exps = exps[range(N), y]

    loss = -np.log(true_exps / np.sum(exps, axis=1))
    loss = np.sum(loss) / N

    mask = np.zeros_like(X)
    mask = exps / np.sum(exps,axis=1)[:, None]
    mask[range(N), y] -= 1.0
    dx = mask / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
