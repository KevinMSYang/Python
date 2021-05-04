
# coding: utf-8

# <b>Homework 06 - Multilayer Perceptron
# >Kevin Yang 50541650

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


X = np.array([[0,0],[0,1],[1,0],[1,1]])
r = np.array([0,1,1,0])


# In[3]:


np.random.seed(2019)
w0 = np.random.normal(0, 0.5, size=(2,3))
w1 = np.random.normal(0, 0.5, size=(1,3))


# In[4]:


def sigmoid(a):
    return 1/(1+np.exp(-a))


# In[5]:


def sigmoid_deriv(a):
    return a*(1-a)


# In[6]:


def augment_for_bias(m):
    if len(m.shape) >1:
        mhat = np.ones((m.shape[0],m.shape[1]+1))
        mhat[:,1:] *= m
    else:
        mhat = np.concatenate(([1],m))
    return mhat


# In[7]:


def forward_pass(X, weights):
    aX = augment_for_bias(X)
    XW0 = w0.dot(aX)
    z = sigmoid(XW0)
    aZ = augment_for_bias(z)
    XW1 = aZ.dot(w1.T)
    y = sigmoid(XW1)
    activations = [z,y]
    return y, activations


# In[8]:


def backward_pass(X, r, weights, activations, learning_rate):
    z, y = activations
    err = r-y
    delta_w1 = err*sigmoid_deriv(y)*z
    delta_b1 = err*sigmoid_deriv(y)
    err_z = delta_w1*weights[1][:,1:]
    aZ = sigmoid_deriv(z).T
    temp = X.dot(aZ)
    delta_w0 = np.multiply(err_z,temp)
    delta_b0 = np.multiply(err_z,sigmoid_deriv(z))
    delta_b0 = delta_b0.reshape(2,)
    weights[0][:,1:] = weights[0][:,1:]+learning_rate*delta_w0
    weights[1][:,1:] = weights[1][:,1:]+learning_rate*delta_w1
    weights[0][:,0] = weights[0][:,0]+learning_rate*delta_b0
    weights[1][:,0] = weights[1][:,0]+learning_rate*delta_b1
    
    return err


# In[9]:


np.random.seed(2019)
w0 = np.random.normal(0,0.5,size=(2,3))
w1 = np.random.normal(0,0.5,size=(1,3))
print("Initial random weights: \n{}\n{}\n".format(w0,w1))


# In[10]:


weights = [w0,w1]
n_epochs = 25000
learning_rate = 0.1


# In[11]:


print("Before training-------")
for i in range(len(X)):
    y, _ = forward_pass(X[i],weights)
    print("{} XOR {} = {} (r = {})".format(X[i][0],X[i][1],y,r[i]))
print("----------------------")


# In[12]:


avg_err_each_epoch = []
for epoch in range(n_epochs):
    log = ""
    epoch_err = []
    for i in range(len(X)):
        y, activations = forward_pass(X[i],weights)
        err = backward_pass(X[i],r[i],weights,activations,learning_rate)
        epoch_err.append(err)
        log += "err {}: {}\n".format(i,err)
    if epoch % 100 == 0 or epoch == n_epochs-1:
        print("After epoch {}:".format(epoch))
        print(log)
        print("Weights:\nw0:\n{}w1:\n{}\n".format(w0.round(4),w1.round(4)))
        for i in range(len(X)):
            y, _ = forward_pass(X[i],weights)
            print("{} XOR {} = {} (r = {})".format(X[i][0],X[i][1],y,r[i]))
        print("")
    avg_err_each_epoch.append(np.mean(np.abs(epoch_err)))
    if avg_err_each_epoch[-1] < 0.1:
        print("Early stopping condition triggered at epoch {}.".format(epoch))
        break


# In[13]:


plt.plot(avg_err_each_epoch)
plt.show()

