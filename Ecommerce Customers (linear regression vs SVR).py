#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# In[14]:


data=pd.read_csv('Ecommerce Customers')
data


# In[6]:


data.info()


# In[7]:


data=data.dropna()


# In[15]:


x=data.drop(columns=['Email','Address','Avatar','Avg. Session Length','Yearly Amount Spent'])
y=np.array(data['Yearly Amount Spent']).reshape(-1,1)


# In[16]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
y_scaled=scaler.fit_transform(y)


# In[46]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_scaled,test_size=0.4,random_state=44)
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[39]:


r_squared=lr.score(x_test,y_test)
print('r_squared score is : ',r_squared)


# In[40]:


y_pred=lr.predict(x_test)


# In[41]:


mean_squared_error(y_test,y_pred)


# In[47]:


scores = cross_val_score(lr, x_scaled, y_scaled, cv=5)

# Print the cross-validation scores
print("Cross-validation scores: {}".format(scores))
print("Mean cross-validation score: {:.2f}".format(scores.mean()))
print("mean squared error :",scores.mean_squared_error())


# In[43]:


svr=SVR(kernel='rbf',C=1.0,epsilon=0.1)
svr.fit(x_train,y_train)


# In[30]:


r_squared=svr.score(x_test,y_test)
print('r_squared score is : ',r_squared)


# In[24]:


y_predd=svr.predict(x_test)


# In[31]:


mean_squared_error(y_test,y_predd)


# we can see that the linear regression model is good than SVC model so lets try to minimize the error between our data in olinear regression model by using 'grid search'

# In[36]:


#Graid Search for linear regression model 

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'copy_X': [True, False], 'fit_intercept': [True, False], 'positive': [True, False]}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(x_train, y_train)

# Get the best hyperparameters and the best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

# Train a new model with the best hyperparameters
lr_best = LinearRegression(**best_params)
lr_best.fit(x_train, y_train)

# Make predictions on the test data
y_predd = lr_best.predict(x_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_predd)
print('MSE:', mse)
print("Best hyperparameters: ", grid_search.best_params_)


# In[ ]:





# In[ ]:






# In[ ]:




