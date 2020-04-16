#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[2]:


numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes) # Hay 1797 digitos representados en imagenes 8x8


# In[3]:


data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[5]:


# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)


# In[6]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[7]:


clf = LogisticRegression(penalty='l1', solver='saga', tol=0.1)
clf.fit(x_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(x_test, y_test)
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)


# In[11]:


coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))

for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8, 8), interpolation='nearest', cmap=plt.cm.RdBu)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)


#run_time = time.time() - t0
#print('Example run in %.3f s' % run_time)
plt.savefig("coeficientes.png")


# In[10]:


y_prediction = clf.predict(x_test)
numeros = np.arange(0,10)    
c_matrix = confusion_matrix(y_test, y_prediction, labels = numeros)
plt.figure(figsize=(10,10))
#plt.title("C from 0.01 to 10")
plt.imshow(c_matrix)
for i in range(10):
    for j in range(10):
    
        plt.text(i, j, "{:.2f}".format( c_matrix[i,j]/np.sum(c_matrix[i,:]) ))
plt.savefig("confusion.png")


# In[ ]:




