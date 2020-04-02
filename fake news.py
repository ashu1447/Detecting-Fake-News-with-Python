#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[79]:


#Read the data
df=pd.read_csv('C:\\Users\\Admin-pc\\Downloads\\news.csv')
#Get head
df.head()


# In[80]:


#Get the labels
labels=df.label
labels.head()


# In[81]:


#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[82]:


#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.99)
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[83]:


#nitialize a PassiveAggressiveClassifier
pa=PassiveAggressiveClassifier(max_iter=100)
pa.fit(tfidf_train,y_train)
#calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[84]:


#confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




