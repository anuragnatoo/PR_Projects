#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# In[3]:


w3 = []
setosa = []  
virginica = []
versicolor = []


# In[4]:


data = pd.read_csv("iris.csv")
print(data.head())
column = ['petal.length', 'sepal.width']
versicolor = data[column][50 : 100].copy()
setosa = data[column][0 : 50].copy()
cla=['variety']
set_class=data[cla][0:100]
print(setosa.head())
print(versicolor.head())


# In[5]:


w1 = []
w2 = []
for index, row in setosa.iterrows():
    w1.append([row[0], row[1]])
w1=np.array(w1)
#print(w1)
print(w1.shape)
for index, row in versicolor.iterrows():
    w2.append([row[0], row[1]])
w2 = np.array(w2)
#print(w2)
print(w2.shape)
y=[]
print(set_class.shape)
for i,ind in set_class.iterrows():
    #print(ind[0])
    if ind[0]=='Setosa':
        y.append(0)
    else:
        y.append(1)
y=np.array(y)
y=y.reshape((100,1))
print(y)
print(y.shape)


# In[6]:


f, ax = plt.subplots(figsize=(7, 7))
c1, c2, c3 = "#3366AA", "#AA3333", 'm'


# In[7]:


ax.scatter(*w1.T, c=c1,s = 10, label = "w1")
ax.legend()
ax.scatter(*w2.T, c=c2, marker="D", label = "w2")
ax.legend()


# In[8]:


x = []
for i in range(len(w1)):
    x.append([w1[i][0], w1[i][1]])
for i in range(len(w2)):
    x.append([w2[i][0], w2[i][1]])
x = np.array(x)
print(x.shape)


# In[9]:


print(x)
print(y)


# In[ ]:





# In[ ]:




