  def nystrom_approximation_fit(X, S, kernel_function, kernel_param):
  """
  This function compute the Nyströmm approximation of a kernel, using only the data indexed by S

  X: numpy array of size (n, d) where n is the number of data and d number of features.
  S: list of selected data indices for the Nystrom approximation.
  kernel_function: a function that compute the kernel matrix, in the same form as the functions of the library sklearn.metrics.pairwise.
  kernel_param: the parameter of the kernel function (the degree if polynomial kernel for example).


  Output:
  mapped_X: numpy array of size (n, |S|)
  inv_K_s: numpy array of size (|S|, |S|) 
  """

  K_1_1 = kernel_function(X[S], X[S], kernel_param)

  # sqrt of kernel matrix on basis vectors
  U, sigma, V = svd(K_1_1)
  sigma = np.maximum(sigma, 1e-12)

  # compute K_{1, 1}^{-1/2}
  normalization_ = np.dot(U / np.sqrt(sigma), V) 

  embedded = kernel_function(X, X[S], kernel_param)

  X_mapped = np.dot(embedded, normalization_.T)
  return X_mapped, normalization_
