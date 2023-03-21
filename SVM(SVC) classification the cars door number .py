#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_selection import SelectKBest,f_regression,f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR,SVC,LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Library for the statistic data vizualisation
import seaborn

get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


data =pd.read_csv('CarPrice_Assignment.csv')
data


# In[36]:


x=data.drop('doornumber',axis=1)
y=data['doornumber']


# In[37]:


one_hot=OneHotEncoder()
x=one_hot.fit_transform(x)


# In[38]:


scaler=StandardScaler(with_mean=False)
x=scaler.fit_transform(x)


# In[39]:


selector=SelectKBest(f_classif,k=4)
x=selector.fit_transform(x,y)


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lo_r=SVC(kernel='rbf',C=0.1,random_state=42,gamma=1)
lo_r.fit(x_train,y_train)


# In[11]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Compute predictions
y_pred = lo_r.predict(x_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,pos_label='four')
recall = recall_score(y_test, y_pred,pos_label='four')
f1 = f1_score(y_test, y_pred,pos_label='four')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


# In[12]:


scores = cross_val_score(lo_r, x, y, cv=5)

# Print the cross-validation scores
print("Cross-validation scores: {}".format(scores))
print("Mean cross-validation score=Accuracy: {:.2f}".format(scores.mean()))


# In[14]:


param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 10]
}

# Create an instance of the SVM classifier


# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=lo_r, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(x_train, y_train)

# Print the best hyperparameters and the corresponding accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)


# In[32]:


y_pred=lo_r.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# accuracy score =0.78\
# accuracy score after using cross validation is 0.85\
# accuracy score after using graid search is 0.866
# 

# 

# In[ ]:





# In[ ]:





# In[ ]:




