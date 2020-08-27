import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

def cost_function(a,b,theta):
	n = len(a)
	sum_fun = np.power(((a*theta.T)-b),2)
	J = np.sum(sum_fun) / (2*n)
	return J

def gradient_descent(a,b,theta,alpha,iterations):
	theta_new = []
	cost = []

	for i in range(iterations):
		h = (a*theta.T)-b			# Error function

		for j in range(len(a)):
			a_error = np.multiply(a[:,j],h)
			# theta_new.append(theta[0,j]-alpha/len(a)*np.sum(a_error[:,j]))
			# theta = theta_new
		
		# cost.append(cost_function(a,b,theta_new))
	return a_error




path = os.getcwd() + '/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population of City in 10,000s', 'Profit in $10,000s'])
data.head()
data.describe()
data.plot.scatter(x='Population of City in 10,000s',y='Profit in $10,000s')
# plt.show()

data.insert(0, 'Ones', 1)
cols = data.shape[1]		# Number of columns
population_train = data.iloc[:,0:cols-1]
profit_target = data.iloc[:,cols-1:cols]

a = np.matrix(population_train.values)
b = np.matrix(profit_target.values) 
theta = np.matrix(np.array([0,0]))


alpha = 0.01
iterations = 1000

print(gradient_descent(a, b, theta, alpha, iterations))



