#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[3]:


data=pd.read_csv('advertising.csv')
data.head()


# In[5]:


data.describe().T


# In[7]:


data.shape


# In[8]:


data.isna().sum()/data.shape[0]*100


# In[9]:


data.dtypes


# In[ ]:





# In[10]:


x=data['TV']
y=data['Sales']


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=100)


# In[13]:


import statsmodels.api as sm


# In[14]:


#add constant to get an intercept
x_train_sm=sm.add_constant(x_train)


# In[17]:


reg=sm.OLS(y_train,x_train_sm).fit()


# In[19]:


reg.params


# # y=6.780417+0.055639*TV

# In[22]:


plt.scatter(x_train,y_train)
plt.plot(x_train,6.780417+0.055639*x_train,'r')
plt.show()


# In[23]:


#lets see the error terms 


# In[24]:


y_train_pred=reg.predict(x_train_sm)


# In[25]:


res=(y_train-y_train_pred)


# In[27]:


fig=plt.figure()
sns.displot(res,bins=15)
fig.suptitle('Error Terms',fontsize=16)
plt.xlabel('y_train-y_train_pred')
plt.show()


# In[30]:


x_test_sm=sm.add_constant(x_test)
y_pred=reg.predict(x_test_sm)


# In[33]:


from sklearn.metrics import mean_squared_error


# In[34]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[37]:


plt.scatter(x_test,y_test)
plt.plot(x_test,6.780417+0.055639*x_test,'r')


# In[ ]:




