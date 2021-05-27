#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# In[2]:


setosa = []  
versicolor = []


# In[3]:


data = pd.read_csv("iris.csv")
print(data.head())
column = ['petal.length', 'sepal.width']
versicolor = data[column][50 : 100].copy()
setosa = data[column][0 : 50].copy()
cla=['variety']
set_class=data[cla][0:100]
print(setosa.head())
print(versicolor.head())


# In[4]:


w1 = []
w2 = []
for index, row in setosa.iterrows():
    w1.append([row[0], row[1]])
w1=np.array(w1)
print(w1)
print(w1.shape)
for index, row in versicolor.iterrows():
    w2.append([row[0], row[1]])
w2 = np.array(w2)
print(w2)
print(w2.shape)


# In[5]:


data1 = []
for i in range(len(w1)):
    data1.append([1, w1[i][0], w1[i][1]])
    data1.append([-1, -w2[i][0], -w2[i][1]])
data1 = np.array(data1)
print(data1.shape)


# In[6]:


data2 = []
for i in range(len(w2)):
    data2.append([1, w2[i][0], w2[i][1]])
    data2.append([-1, -w1[i][0], -w1[i][1]])
data2 = np.array(data2)
print(data2.shape)


# In[7]:


lr=0.01
k=0
Q=0
a = np.array([0, 0, 0])
missclasified_vectors = True
epoch = 1


# In[8]:


while(((missclasified_vectors) or (not(k == 0))) and (epoch < 10)):
        if(k == 0):
            print("\n\n\nIteration : ", epoch)
            epoch = epoch + 1
            missclasified_vectors = False
        print("Data index : ", k)
        print("weights : ", a)
        print("Data point : ", np.array(data1[k]))
        dot_product = np.dot(a, data1[k])
        print("The dot product is : ", dot_product)
        if(dot_product <= 0):            
            a = a + lr * data1[k]
            missclasified_vectors = True
        k = (k + 1) % (len(data1))
print("***************Perceptron*****************")
print("weights are : ", a)
x_vec = np.linspace(0, 10, 5)
y_vec = ((a[1] * x_vec) + a[0])/a[2]
y_vec = -y_vec 
f, ax = plt.subplots(figsize=(7, 7))
c1, c2, c3 = "#3366AA", "#AA3333", 'm'
ax.scatter(*w1.T, c=c1,s = 10, label = "w1")
ax.legend()
ax.scatter(*w2.T, c=c2, marker="D", label = "w2")
ax.legend()
ax.set_title('Perceptron')
plt.plot(x_vec, y_vec, 'g--')
plt.show()