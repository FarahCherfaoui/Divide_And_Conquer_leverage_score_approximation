import numpy as np
from sklearn.metrics.pairwise import *
import sys
from scipy.linalg import svd
from math import *
import os
import scipy.linalg as spl
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from DAC import *
from nystrom_approximation import *





"""
this code allows us to test the DAC algorithm, in the context of 
active learning. 
We use DAC to compute an approximation of the leverage score. 
We then use these scores to sample subsets of data that we use 
to train a model (here a SVM).

we repeat this experiment for several subset sizes and plot the error.
This experiment can be used to generate figure 8 of the submitted paper.
"""



# dowlowd the data
X, y = fetch_openml('kc1', return_X_y = True)
n = X.shape[0]

#compute approximated leverage scores using DAC algorithm
#fixe hyperparameters
kernel_parameter = 1
lambda_ = 1
nb_repeat = 20




# we fixe the budget of label as a percentage of the total dataset size.
pr_label = [0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
test_score = np.zeros(len(pr_label))


for j in range(nb_repeat):
	
	# compute the approximated leverage scores using the different methods.
	# In the DAC case, the size of each sub-matrix is set to \sqrt{n}. 
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=j)
	n_tr = X_train.shape[0]
	sample_size = int(sqrt(n_tr))
	approximated_ls_DAC = DAC(X_train, lambda_, sample_size, rbf_kernel, kernel_parameter)


	
	for (i, pr) in zip(range(len(pr_label)), pr_label):
		#sample pr% of data proportionally to these scores, and compute Nyström approximation using the sampled data
		selected = np.random.choice(n_tr, size=int(pr*n_tr), replace=False, p=approximated_ls_DAC/np.sum(approximated_ls_DAC))
		
		# train the model using only the selected data point
		model = SVC(kernel = 'rbf', gamma=kernel_parameter).fit(X_train[selected], y_train[selected])
		test_score[i] += model.score(X_test, y_test)

print("test_score: ", test_score)
test_score /= nb_repeat
print("test_score: ", test_score)




# plot the error
plt.plot([n*pr for pr in pr_label], test_score, label = 'DAC')
plt.xlabel('label budget')
plt.ylabel(r'score')
plt.legend()
plt.show()
