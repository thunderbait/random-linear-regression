import numpy as np 
from sklearn import datasets 
import seaborn.apionly as sns 
%matplotlib inline 
import matplotlib.pyplot as plt 

sns.set(style='whitegrid', context='notebook')

iris2 = sns.load_dataset('iris')

def covariance(X,Y): 
	xhat=np.mean(X) yhat=np.mean(Y) epsilon =0
for x,y in zip(X,Y):
	epsilon=epsilon+(x-xhat)*(y-yhat)

	return epsilon/(len(X)-1)

print(covariance([1,3,4],[1,0,2]))
print(np.cov([1,3,4], [1,0,2]))