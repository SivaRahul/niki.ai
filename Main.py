
# coding: utf-8

# In[4]:

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt


# In[5]:

df = pd.read_csv("D:\\niki.ai\\LabelledData.txt",sep=",,,",header=None ,names=['question','type'])


# In[6]:

import re
df['question'] = df['question'].apply(lambda x: x.lower())
df['question'] = df['question'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))


# In[7]:

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


# In[8]:

stemmer = SnowballStemmer('english').stem
def stem_tokenize(text):
    return [stemmer(i) for i in word_tokenize(text)]
val_split=0.30
vectorizer = CountVectorizer(analyzer='word',lowercase=True,tokenizer=stem_tokenize)


# In[9]:

X_train = vectorizer.fit_transform(df.question.values)
labels = df['type']
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
labels = labels[indices]
x_train = X_train
y_train = labels
x_test = raw_input("Enter the quesion you want to classify?")
x_test = x_test.lower()
x_test = re.sub('[^a-zA-z0-9\s]','',x_test)
x_test=[x_test]


# In[10]:

vectorizer_test = CountVectorizer(analyzer='word',lowercase=True,tokenizer=stem_tokenize)
x_test_vect= vectorizer.transform(x_test)


# In[11]:

from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
clf_svm.fit(x_train,y_train)
preds_svm = clf_svm.predict(x_test_vect)
print preds_svm


# In[ ]:



