
# coding: utf-8

# In[113]:

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt


# In[114]:

df = pd.read_csv("D:\\niki.ai\\LabelledData.txt",sep=",,,",header=None ,names=['question','type'])


# In[115]:

df


# In[116]:

type(df['question'])


# In[117]:

import re
df['question'] = df['question'].apply(lambda x: x.lower())
df['question'] = df['question'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))


# In[118]:

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import wordnet as wn


# In[119]:

stemmer = SnowballStemmer('english').stem
def stem_tokenize(text):
    return [stemmer(i) for i in word_tokenize(text)]
val_split=0.30
vectorizer = CountVectorizer(analyzer='word',lowercase=True,tokenizer=stem_tokenize)


# In[120]:

X_train = vectorizer.fit_transform(df.question.values)
labels = df['type']
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
labels = labels[indices]
nb_validation_samples = int(val_split * X_train.shape[0])

x_train = X_train[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = X_train[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

clf_nb = MultinomialNB()
clf_nb.fit(x_train,y_train)

preds_nb = clf_nb.predict(x_val)
print(classification_report(preds_nb,y_val))
print("Accuracy :",clf_nb.score(x_val,y_val))


# In[121]:

type(df['question'])


# In[122]:

from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
clf_svm.fit(x_train,y_train)
preds_svm = clf_svm.predict(x_val)
print(classification_report(preds_svm,y_val))
print("Accuracy :",clf_svm.score(x_val,y_val))


# In[ ]:




# In[ ]:




# In[ ]:



