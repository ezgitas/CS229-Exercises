import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

path = os.getcwd() + '/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2','Admitted'])
data.head()
data.describe()
data.plot.scatter(x='Population of City in 10,000s',y='Exam 2')
#plt.show()