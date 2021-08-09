def DAC(X, lambda_, sample_size, kernel_function, kernel_param):
  """
  This function computes an approximation of the ridge leverage score, using a divide and conquer strategy.

  X: numpy array of size (n, d) where n is the number of data and d number of features.
  lambda_: regularisation term.
  sample_size: size of sub-matrix.
  kernel_function: a function that compute the kernel matrix, in the same form as the functions of the sklearn.metrics.pairwise library.
  kernel_param: the parameter of the kernel function (the degree if polynomial kernel for example).

  """

  if (lambda_ < 0 or kernel_param < 0):
    print("lambda and kernel parameter have to be positive")
    exit(1)

  n = X.shape[0]
  ind = np.arange(n)
  np.random.shuffle(ind)
  approximated_ls = np.zeros((n))


  for l in range(0, ceil(n/sample_size)):
    # sample a subset of data
    true_sample_size = min(sample_size, n - l*sample_size)
    temp_ind = ind[l*sample_size: l*sample_size + true_sample_size]
    
    # compute the kernel matrix using the subset of selected data
    K_S = kernel_function(X[temp_ind], X[temp_ind], kernel_param)
    
    # compute the approximated leverage score by inverting the small matrix
    approximated_ls[temp_ind] = np.sum(K_S * np.linalg.inv(K_S + lambda_ * np.eye(true_sample_size)) , axis = 1)

  return approximated_ls
