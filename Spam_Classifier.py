#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download()


# In[2]:


# importing the Dataset
import pandas as pd


# In[3]:


messages = pd.read_csv('SMSSpamCollection',sep = '\t',
                      names = ['label','message'])


# In[4]:


messages.head()


# In[5]:


#Data cleaning and preprocessing
import re


# In[6]:


nltk.download('stopwords')


# In[9]:


from nltk.corpus import stopwords


# In[10]:


from nltk.stem.porter import PorterStemmer


# In[11]:


ps = PorterStemmer()


# In[12]:


corpus = []


# In[13]:


for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[15]:


print(corpus)


# In[16]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


cv = CountVectorizer(max_features = 2500)


# In[18]:


X = cv.fit_transform(corpus).toarray()


# In[19]:


print(X)


# In[23]:


y = pd.get_dummies(messages['label'])
print(y)


# In[29]:


y = y.iloc[:,1].values


# In[30]:


print(y)


# In[31]:


#Train test split
from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state=0)


# In[33]:


# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB


# In[34]:


spam_detect_model = MultinomialNB().fit(X_train,y_train)


# In[35]:


y_pred=spam_detect_model.predict(X_test)


# In[36]:


print(y_test)


# In[37]:


print(y_pred)


# In[43]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[41]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[42]:


cr = classification_report(y_test,y_pred)
cr


# In[45]:


acc = accuracy_score(y_test,y_pred)
acc


# In[ ]:




