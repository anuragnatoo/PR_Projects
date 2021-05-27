#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


def perceptron():
    w1 = np.array([[1, 1, 6], [1, 7, 2], [1, 8, 9], [1, 9, 9]])
    w2 = np.array([[1, 2, 1], [1, 2, 2], [1, 2, 4], [1, 7, 1]])
    data = np.concatenate([w1, -w2], axis = 0)
    print(data)
    w1 = np.array([[1, 6], [7, 2], [8, 9], [9, 9]])
    w2 = np.array([[2, 1], [2, 2], [2, 4], [7, 1]])
    total_epoch = [0, 0]
    for learning_rate in [0.1]:
        k = 0
        Q = 0                           #criterian theta
        a = np.array([0.1, 0.1, 0.1])            #weights
        mv = True
        epoch = 1
        while((mv) or not(k == 0)):
            if(k == 0):
                print("\n\n\nEpoch : ", epoch)
                epoch = epoch + 1
                mv = False
            print("Data index : ", k)
            print("weights : ", a)
            print("Data point : ", np.array(data[k]))
            dp = np.dot(a, data[k])
            print("The dot product is : ", dp)
            if(dp <= 0):            
                a = a + learning_rate * data[k]
                mv = True
            k = (k + 1) % (len(data))
    print("The final weight vector is : ", a)
    f, ax = plt.subplots(figsize=(7, 7))
    c1, c2 = "#3366AA", "#AA3333"
    ax.scatter(*w1.T, c=c1,s = 10, label = "w1")
    ax.legend()
    ax.scatter(*w2.T, c=c2, marker="D", label = "w2")
    ax.legend()
    x_vec = np.linspace(-2, 10, 5)
    y_vec = ((a[1] * x_vec) + a[0])/a[2]
    y_vec = -y_vec   
    plt.plot(x_vec, y_vec, 'r--')

    b = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    print(data.shape)
    a = np.linalg.lstsq(data, b.transpose())
    a = np.array(a[0])
    x_vec = np.linspace(-2, 10, 5)
    y_vec = ((a[1] * x_vec) + a[0])/a[2]
    y_vec = -y_vec   
    plt.plot(x_vec, y_vec, 'b--')
    plt.show()


# In[3]:


perceptron()


# In[ ]:




