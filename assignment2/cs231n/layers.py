from builtins import range
import numpy as np


def affine_forward(X, W, bias):
    """Computes the forward pass for an affine (fully connected) layer.

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
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    X, W, bias = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    M = bias.shape[0]

    X_rows = np.copy(X).reshape(N, -1)
    X_rows = np.hstack((X_rows, np.ones((N, 1)))) # N x D+1
    W_bias = np.vstack((W, bias.T)) # D+1 x M

    dWb = np.dot(X_rows.T, dout) # D+1 x M
    dw = dWb[0:-1, :] # D x M
    db = dWb[-1, :].reshape(M, 1) # M x 1

    dX_rows = np.dot(dout, W_bias.T) # N x D+1
    dx = dX_rows[:, 0:-1] # N x D
    dx = dx.reshape(X.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, X = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.copy(dout)
    mask = np.float64(X > 0)
    dx *= mask

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(X, y):
    """Computes the loss and gradient for softmax classification.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]

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


def batchnorm_forward(X, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

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
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = X.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=X.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=X.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        batch_mean = np.mean(X, axis=0) # 1 x D
        X_centralized = X - batch_mean # N x D

        batch_variance = np.mean(X_centralized**2, axis=0) # 1 x D
        X_normalized = X_centralized / np.sqrt(batch_variance + eps)

        out = X_normalized * gamma + beta

        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_variance

        bn_param["running_mean"] = running_mean
        bn_param["running_var"] = running_var

        cache = {}
        cache["gamma"] = gamma
        cache['X_centralized'] = X_centralized
        cache['batch_variance'] = batch_variance
        cache['eps'] = eps
        cache["X_norm"] = X_normalized

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        X_normalized = (X - running_mean) / np.sqrt(running_var + eps)
        out = X_normalized * gamma + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma = cache["gamma"]
    X_norm = cache["X_norm"]
    X_centralized = cache['X_centralized']
    batch_variance = cache['batch_variance']
    eps = cache['eps']
    N = X_norm.shape[0]


    dX_norm = dout * gamma # N x D

    dVariance = np.sum(dX_norm * X_centralized, axis=0) * \
                        np.power(batch_variance + eps, -1.5) / -2 # 1 x D

    dMean = np.sum(dX_norm / -np.sqrt(batch_variance + eps), axis=0) + \
            dVariance * np.sum(X_centralized, axis=0) * -2 / N           # 1 x D

    dx = dX_norm / np.sqrt(batch_variance + eps) + dVariance * 2 * X_centralized / N + \
          dMean / N   # N x D

    dgamma = np.sum(dout * X_norm, axis=0) # 1 x D

    dbeta = np.sum(dout, axis=0)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma = cache['gamma']
    X_norm = cache['X_norm']
    batch_variance = cache['batch_variance']
    eps = cache['eps']
    N = len(X_norm)

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(X_norm * dout, axis=0)
    dx = (dout - (dgamma * X_norm + dbeta) / N) * gamma / np.sqrt(batch_variance + eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(X, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # from IPython.core.debugger import set_trace
    # set_trace()

    batch_mean = np.mean(X, axis=1) # N x ,
    X_centralized = X - batch_mean[:, None] # N x D

    batch_variance = np.mean(X_centralized**2, axis=1) # N x ,
    X_normalized = X_centralized / np.sqrt(batch_variance + eps)[:, None]

    out = X_normalized * gamma + beta

    cache = {}
    cache["gamma"] = gamma
    cache['X_centralized'] = X_centralized
    cache['batch_variance'] = batch_variance
    cache['eps'] = eps
    cache["X_norm"] = X_normalized

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # gamma = cache['gamma']
    # X_norm = cache['X_norm']
    # batch_variance = cache['batch_variance']
    # eps = cache['eps']
    # N = len(X_norm)

    gamma = cache["gamma"]
    X_norm = cache["X_norm"]
    X_centralized = cache['X_centralized']
    batch_variance = cache['batch_variance']
    eps = cache['eps']
    N = X_norm.shape[1]

    dX_norm = dout * gamma # N x D

    # from IPython.core.debugger import set_trace
    # set_trace()

    dVariance = np.sum(dX_norm * X_centralized, axis=1) * \
                        np.power(batch_variance + eps, -1.5) / -2 # 1 x N

    dMean = np.sum(dX_norm / -np.sqrt(batch_variance + eps)[:, None], axis=1) + \
            dVariance * np.sum(X_centralized, axis=1) * -2 / N           # 1 x N

    dx = dX_norm / np.sqrt(batch_variance + eps)[:, None] + dVariance[:, None] * 2 * X_centralized / N + \
          dMean[:, None] / N   # N x D

    dgamma = np.sum(dout * X_norm, axis=0) # 1 x D

    dbeta = np.sum(dout, axis=0)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(X, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*X.shape) < p) / p
        out = X * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = X

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(X.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # p = dropout_param['p']
        dx = dout * mask# * p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(X, W, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    matrWidth = X.shape[3]
    matrHeight = X.shape[2]
    filtShape = W.shape[2]
    pad = conv_param['pad']
    stride = conv_param['stride']
    numberOfFilt = W.shape[0]
    numberOfImage = X.shape[0]
    numberOfChanels = X.shape[1]

    #  (W−F+2P)/S+1
    outWidth = (matrWidth - filtShape + 2 * pad) // stride + 1
    outHeight = (matrHeight - filtShape + 2 * pad) // stride + 1

    out = np.zeros((numberOfImage, numberOfFilt, outHeight, outWidth)) 

    # Создание новой матрицы с отступом и вставка
    newMatr = np.zeros((numberOfImage, numberOfChanels, matrHeight + pad * 2, matrWidth + pad * 2))
    newMatr[:, :, pad:newMatr.shape[2] - pad, pad:newMatr.shape[3] - pad] = X

    # from IPython.core.debugger import set_trace
    # set_trace()

    for indxImage in range(numberOfImage):
      for indxFilt in range(numberOfFilt):
        for numX, i in enumerate(range(0, newMatr.shape[3] - filtShape + 1, stride)):
          for numY, j in enumerate(range(0, newMatr.shape[2] - filtShape + 1, stride)):

            out[indxImage, indxFilt, numY, numX] = np.sum(newMatr[indxImage, :, j:j+filtShape, i:i+filtShape] * \
            W[indxFilt, :, :, :]) + b[indxFilt]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (X, W, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    if len(cache) == 4:
      X, W, b, conv_param = cache
    else:
      X, W, b, conv_param, _ = cache
    pad = conv_param['pad']
    stride = conv_param['stride']

    matrWidth = X.shape[3]
    matrHeight = X.shape[2]
    filtShape = W.shape[2]
    pad = conv_param['pad']
    stride = conv_param['stride']
    numberOfFilt = W.shape[0]
    numberOfImage = X.shape[0]
    numberOfChanels = X.shape[1]

    newMatr = np.zeros((numberOfImage, numberOfChanels, matrHeight + pad * 2, matrWidth + pad * 2))
    newMatr[:, :, pad:newMatr.shape[2] - pad, pad:newMatr.shape[3] - pad] = X

    dw = np.zeros_like(W)
    dx = np.zeros((X.shape[1], X.shape[2], X.shape[3]))
    # dxWithPad = np.zeros((newMatr.shape[1], newMatr.shape[2], newMatr.shape[3]))
    dxWithPad = np.zeros_like(newMatr)
    db = np.zeros_like(b)

    for indxImage in range(numberOfImage):
      for indxFilt in range(numberOfFilt):
        for indxChanel in range(numberOfChanels):
          for numX, i in enumerate(range(0, newMatr.shape[3] - filtShape + 1, stride)):
            for numY, j in enumerate(range(0, newMatr.shape[2] - filtShape + 1, stride)):
              
              dw[indxFilt, indxChanel] = dw[indxFilt, indxChanel] + \
              newMatr[indxImage, indxChanel, j:j+filtShape, i:i+filtShape] * \
              dout[indxImage, indxFilt, numY, numX]

              dxWithPad[indxImage, indxChanel, j:j+filtShape, i:i+filtShape] = \
              dxWithPad[indxImage, indxChanel, j:j+filtShape, i:i+filtShape] + \
              W[indxFilt, indxChanel] * dout[indxImage, indxFilt, numY, numX]

    # dxWithPad = np.sum(dxWithPad, axis=0)

    # from IPython.core.debugger import set_trace
    # set_trace()

    db = np.sum(np.sum(np.sum(dout, axis=0), axis=1), axis=1)

    dx = dxWithPad[:, :, pad:newMatr.shape[2] - pad, pad:newMatr.shape[3] - pad]
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(X, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    poolH = pool_param['pool_height']
    poolW = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = X.shape

    outH = (H - poolH) // stride + 1
    outW = (W - poolW) // stride + 1

    out = np.zeros((N, C, outH, outW ))

    # from IPython.core.debugger import set_trace
    # set_trace()

    for numY, i in enumerate(range(0, H, stride)):
      for numX, j in enumerate(range(0, W, stride)):

        out[:, :, numY, numX] = np.max(X[:, :, i:i+poolH, j:j+poolW], axis=(2, 3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (X, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X, pool_param = cache
    poolH = pool_param['pool_height']
    poolW = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros_like(X)

    N, C, H, W = X.shape

    for img in range(N):
      for chanel in range(C):
        for numY, i in enumerate(range(0, H, stride)):
          for numX, j in enumerate(range(0, W, stride)):
          
            indx = np.unravel_index(np.argmax(X[img, chanel, i:i+poolH, j:j+poolW]), (2, 2))
            dx[img, chanel, i + indx[0], j + indx[1]] = dout[img, chanel, numY, numX]

    # from IPython.core.debugger import set_trace
    # set_trace()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(X, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

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

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = X.shape
    X_T = X.transpose((0,2,3,1))  
    X_flat = X_T.reshape(-1, C)  
    out, cache = batchnorm_forward(X_flat, gamma, beta, bn_param)
    out = out.reshape((N,H,W,C)).transpose(0,3,1,2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_T = dout.transpose((0,2,3,1))
    dout_flat = dout_T.reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_flat, cache)
    dx = dx.reshape((N,H,W,C)).transpose(0,3,1,2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x = x.reshape((N*G, C//G*H*W))
    x = x.T

    sample_mean = np.mean(x, axis=0)  # (N*G,)
    sample_var = np.var(x, axis=0)
    var_corr_root = np.sqrt(sample_var + eps)

    x_norm = (x - sample_mean)/var_corr_root
    x_norm = x_norm.T.reshape((N,C,H,W))

    out = gamma*x_norm + beta
    cache = (gamma, x_norm, var_corr_root, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma, x_norm, var_corr_root, G = cache

    # Set keepdims=True to make dbeta and dgamma's shape be (1, C, 1, 1)
    dgamma = np.sum(x_norm * dout, axis=(0,2,3), keepdims=True)  # sum((N, D) * (N, D)) = (D,)
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)  # (D,)

    N, C, H, W = x_norm.shape
    x_norm = x_norm.reshape((N*G,C//G*H*W)).T

    dxnorm = gamma * dout  # (D,) * (N, D) = (N, D)
    dxnorm = dxnorm.reshape((N*G,C//G*H*W)).T

    # NOTE: dimensions in the comments are inverted
    dmult1 = 1 / var_corr_root * dxnorm  # (N, D)
    dmult2 = np.sum(x_norm * var_corr_root * dxnorm, axis=0)  # (D,)

    dmult2 = -1 / var_corr_root**2 * dmult2  # (D,)

    dmult2 = 0.5 / var_corr_root * dmult2  # (D,)

    N_i, D_i = x_norm.shape
    dmult2 = 1 / N_i * np.ones((N_i, D_i)) * dmult2  # (N, D)

    dmult2 = 2 * x_norm * var_corr_root * dmult2  # (N, D)

    dx = dmult1 + dmult2  # (N, D)

    dterm1 = 1 * dx  # (N, D)
    dterm2 = -1 * np.sum(dx, axis=0)  # (D,)

    dterm2 = 1/N_i * np.ones((N_i, D_i)) * dterm2  # (N, D)

    dx = dterm1 + dterm2  # (N, D)
    dx = dx.T.reshape((N,C,H,W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
