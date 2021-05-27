#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return np.multiply(x, 1-x)
def tanh(x):
    num1=np.exp(x)-np.exp(-1*x)
    den1=np.exp(x)+np.exp(-1*x)
    return num1/den1
def tanh_d(x):
    return (1-(tanh(x)*tanh(x)))


# In[2]:


class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.array([[0.3,0.1,0.2],[0.2,0.1,0.2]])
        #print(self.weights1.shape)
        self.weights2   = np.array([[0.5,0.1,0.2],[0.4,0.1,0.2]])
        #print(self.weights2.shape)
        self.y          = y
        #print(y.shape)
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.weights1, self.input))
        #print(self.layer1.shape)
        self.layer1=np.append(1,self.layer1).reshape(3,1)
        #print(self.layer1)
        self.output = sigmoid(np.dot(self.weights2, self.layer1))
        #print(self.output.shape)
        #print(self.output)
        
    def feedforward2(self):
        self.layer1 = tanh(np.dot(self.weights1, self.input))
        #print(self.layer1.shape)
        self.layer1=np.append(1,self.layer1).reshape(3,1)
        #print(self.layer1)
        self.output = tanh(np.dot(self.weights2, self.layer1))
        #print(self.output.shape)
        #print(self.output)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(((self.y - self.output)*sigmoid_derivative(self.output)),self.layer1.T)
        #print(d_weights2.shape)
        a=np.dot(self.weights2.T,(self.y-self.output)*sigmoid_derivative(self.output))*sigmoid_derivative(self.layer1)
        #print(a.shape)
        self.input=np.delete(self.input, (0), axis=0)
        #print(self.input.shape)
        #print(self.input)
        d_weights1 = np.multiply(self.input,a.T)
        #print(d_weights1.shape)
        self.input=np.append(1,self.input).reshape(3,1)
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    def backprop2(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(((self.y - self.output)*tanh_d(self.output)),self.layer1.T)
        #print(d_weights2.shape)
        a=np.dot(self.weights2.T,(self.y-self.output)*tanh_d(self.output))*tanh_d(self.layer1)
        #print(a.shape)
        self.input=np.delete(self.input, (0), axis=0)
        #print(self.input.shape)
        #print(self.input)
        d_weights1 = np.multiply(self.input,a.T)
        #print(d_weights1.shape)
        self.input=np.append(1,self.input).reshape(3,1)
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


# In[3]:


X = np.array([1.0,0.1,0.2]).reshape(3,1)
y = np.array([0.4,0.3]).reshape(2,1)
nn = NeuralNetwork(X,y)
for i in range(1):
    nn.feedforward()
    nn.backprop()
    print("Iteration-1:")
    print(nn.output)
for i in range(11):
    nn.feedforward2()
    nn.backprop2()
    print("Iteration-",i+2,":")
    print(nn.output)


# ### Convergence is achieved after 12 iterations with Tanh activation function

# In[ ]:




