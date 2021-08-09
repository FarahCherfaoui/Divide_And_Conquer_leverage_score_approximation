# Revisiting-the-Three-Sample-Problem-and-Nystr-m-Approximation-from-Discrete-RKHSs-

Python code implementing the Divide And Conquer (DAC) method for approximating the ridge leverage score proposed in the paper: Revisiting the Three-Sample Problem and Nyström Approximation from Discrete RKHSs (submitted to Nips 2021, number 10610).

## Example
Here we present an example of Nyström approximation for a random data matrix, using an RBF kernel.

import numpy as np  
from sklearn.metrics.pairwise import rbf_kernel   
from math import *  

#generate random matrix and compute the kernel matrix  
n, d = 100, 2  
X = np.random.normal(size=(n, d))  

#compute approximated leverage scores using DAC algorithm  
#the size of each sub-matrix is equal to \sqrt{n}  
kernel_parameter = 1  
lambda_ = 1  
sample_size = int(sqrt(n))  
approximated_leverage_scores = DAC(X, lambda_, sample_size, rbf_kernel, kernel_parameter)  

#sample 10% of data proportionally to these scores  
p = approximated_leverage_scores/np.sum(approximated_leverage_scores)  
selected = np.random.choice(n, size=int(0.1*n), replace=False, p=p)  

#compute Nyström approximation suing the sampled data  
X_mapped, inv_K_s = kernel_approximation_fit(X, selected, rbf_kernel, kernel_parameter)  
approximated_K = np.dot(X_mapped, X_mapped.T)  
