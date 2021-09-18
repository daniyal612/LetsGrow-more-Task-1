#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[2]:


#Read the dataset 

iris=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                 names=["sepal_len","sepal_width","petal_len","petal_width","class"])


# In[6]:


#data understanding and preparation
iris.head()


# In[7]:


#checking if there is any null values
iris.isnull()


# In[10]:


#describing data
iris.describe()


# In[12]:


#shaping data
iris.shape


# In[13]:


#Info = Gives summary of data
iris.info()


# In[14]:


#Data Analysis
iris['class'].value_counts()


# In[16]:


iris.columns


# In[17]:


iris.values


# In[19]:


X=iris.iloc[:,:4]


# In[20]:


X.head()


# In[21]:


y=iris.iloc[:,-1]


# In[22]:


y.head()


# In[26]:


#Data Normalization
X = preprocessing.StandardScaler().fit_transform(X)


# In[29]:


X[0:5]


# In[31]:


#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=1)
y_test.shape


# In[32]:


#Training and Predicting
knnmodel=KNeighborsClassifier(n_neighbors=3)


# In[33]:


knnmodel.fit(X_train,y_train)


# In[34]:



y_predict1=knnmodel.predict(X_test)


# In[35]:


#Accuracy

from sklearn.metrics import accuracy_score


# In[36]:


acc=accuracy_score(y_test,y_predict1)


# In[37]:


acc


# In[38]:


#Confusion Metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values,y_predict1)


# In[39]:


cm


# In[40]:


cm1=pd.DataFrame(data=cm,index=['setosa','versicolor','virginica'],columns=['setosa','versicolor','virginica'])


# In[41]:


cm1


# In[42]:


#Output Visualization
prediction_output=pd.DataFrame(data=[y_test.values,y_predict1],index=['y_test','y_predict1'])


# In[43]:


prediction_output.transpose()


# In[44]:


prediction_output.iloc[0,:].value_counts()


# In[45]:


#value of K
Ks=50
mean_acc=np.zeros((Ks-1))


#train and predict
for n in range(1,Ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1]=accuracy_score(y_test,yhat)


# In[46]:


print(mean_acc)


# In[47]:



print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# In[49]:


#Plotting
plt.plot(range(1,Ks),mean_acc,'g')
plt.legend(('Accuracy '))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[ ]:




