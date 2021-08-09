# Divide And Conquer (DAC) method for approximating the ridge leverage score

Python code implementing the Divide And Conquer (DAC) method for approximating the ridge leverage score proposed in the paper: Revisiting the Three-Sample Problem and Nyström Approximation from Discrete RKHSs (submitted to Nips 2021, number 10610).

The python version used is: 3.7.2

## Usage of the DAC function
```python
DAC(X, lambda_, sample_size, kernel_function, kernel_param):
```
### Input
* X: numpy array of size (n, d) where n is the number of data and d number of features.  
* lambda_: regularisation term.  
* sample_size: size of sub-matrix.  
* kernel_function: a function that compute the kernel matrix, in the same form as the functions of the library sklearn.metrics.pairwise.  
* kernel_param: the parameter of the kernel function (the degree if polynomial kernel for example).  
### Output
* l: a vector of size n, containing the approximations of the leverage scores.


## Example
We present here an example of Nyström approximation for a random data matrix, using an RBF kernel.  

```python
import numpy as np  
from sklearn.metrics.pairwise import rbf_kernel   
from math import *  

#generate random matrix
n, d = 100, 2  
X = np.random.normal(size=(n, d))  

#compute approximated leverage scores using DAC algorithm  
#the size of each sub-matrix is set to \sqrt{n}  
kernel_parameter = 1  
lambda_ = 1  
sample_size = int(sqrt(n))  
approximated_leverage_scores = DAC(X, lambda_, sample_size, rbf_kernel, kernel_parameter)  

#sample 10% of data proportionally to these scores  
p = approximated_leverage_scores/np.sum(approximated_leverage_scores)  
selected = np.random.choice(n, size=int(0.1*n), replace=False, p=p)  

#compute Nyström approximation using the selected data  
X_mapped, inv_K_s = nystrom_approximation_fit(X, selected, rbf_kernel, kernel_parameter)  
approximated_K = np.dot(X_mapped, X_mapped.T)  ```
