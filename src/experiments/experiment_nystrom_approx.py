import numpy as np
from sklearn.metrics.pairwise import *
import sys
from scipy.linalg import svd
from math import *
import os
import scipy.linalg as spl
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from DAC import *
from nystrom_approximation import *





"""
this code allows us to test the DAC algorithm, in the context of 
Nystrom approximation. 
We use DAC to compute an approximation of the leverage score. 
We then use these scores to sample subsets of data that we use 
to compute the Nystrom approximation.

we repeat this experiment for several subset sizes and plot the error.
This experiment can be used to generate figures 6 and 7 of the submitted paper.
"""



# dowlowd the data
# ~ X, y = fetch_openml('covertype', return_X_y = True)
X, y = fetch_openml('kc1', return_X_y = True)
n = X.shape[0]

#compute approximated leverage scores using DAC algorithm
#fixe hyperparameters
kernel_parameter = 1
lambda_ = 1
nb_repeat = 5

# compute the approximated leverage scores using the different methods.
# In the DAC case, the size of each sub-matrix is set to \sqrt{n}. 
sample_size = int(sqrt(n))
approximated_ls_DAC = DAC(X, lambda_, sample_size, rbf_kernel, kernel_parameter)




# we fixe the size of the subsets as a percentage of the total dataset size.
pr_nystrom_data = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]
spectral_error = np.zeros(len(pr_nystrom_data))


for _ in range(nb_repeat):
	# sample a subset of 10 000 data point to evaluate the nystrom approximation
	if (n > 10000):
		ind = np.random.choice(n, size=int(10000), replace=False, p=approximated_leverage_scores/np.sum(approximated_leverage_scores))
	else:
		ind = list(range(n))
	K = rbf_kernel(X[ind], X[ind], kernel_parameter)
	
	for (i, pr) in zip(range(len(pr_nystrom_data)), pr_nystrom_data):
		#sample pr% of data proportionally to these scores, and compute Nyström approximation using the sampled data
		selected = np.random.choice(n, size=int(pr*n), replace=False, p=approximated_ls_DAC/np.sum(approximated_ls_DAC))
		X_mapped, inv_K_s = nystrom_approximation_fit(X, selected, rbf_kernel, kernel_parameter)
		approximated_K = np.dot(X_mapped[ind], X_mapped[ind].T)
		spectral_error[i] += np.linalg.norm(approximated_K - K, 'fro')

spectral_error /= nb_repeat


# plot the error
plt.plot([n*pr for pr in pr_nystrom_data], spectral_error, label = 'DAC')
plt.xlabel('% of data in Nyström approximation')
plt.ylabel(r'$|| \hat{K} - K||_F$')
plt.legend()
plt.show()
