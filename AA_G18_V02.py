
# coding: utf-8

# In[1]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import numpy as np
import pandas as pd


#use all the words in a sentence as features directly, regardless of selection
def preprocess(s):
    return {word: True for word in s.lower().split()}


# In[16]:


#Claim topic classification

#datatype covert to process

#load data
data=pd.read_csv('c_train.csv',sep=',',encoding='utf-8',low_memory=False) 

#panda dataframe convert to list so we can process it later
t1 = data[data['topic']=='Religion does more harm than good']
t1cont = t1['sentence']
list_t1 = t1cont.tolist()

t2 = data[data['topic']=='Science is a major threat']
t2cont = t2['sentence']
list_t2 = t2cont.tolist()

t3 = data[data['topic']=='Newspapers are outdated']
t3cont = t3['sentence']
list_t3 = t3cont.tolist()

#features of each class
t1feats=[(preprocess(str(list_t1[i])), 't1') for i in range(len(list_t1))]
t2feats=[(preprocess(str(list_t2[i])), 't2') for i in range(len(list_t2))]
t3feats=[(preprocess(str(list_t3[i])), 't3') for i in range(len(list_t3))]

#training and testing ratio 
t1cutoff = int(len(t1feats)*0.8)
t2cutoff = int(len(t2feats)*0.8)
t3cutoff = int(len(t3feats)*0.8)


# In[17]:


#training set 
trainfeats = t1feats[:t1cutoff]+t2feats[:t2cutoff]+t3feats[:t3cutoff]
#testing set
testfeats = t1feats[t1cutoff:]+t2feats[t2cutoff:]+t3feats[t3cutoff:]

print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

#naive bayesian model
classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()


# In[90]:


#other models
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(trainfeats)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testfeats))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(trainfeats)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testfeats))

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(trainfeats)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testfeats))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(trainfeats)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testfeats))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(trainfeats)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testfeats))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(trainfeats)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testfeats))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(trainfeats)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testfeats))*100)


# In[14]:


#evidence support or not of a topic classification, same idea same process
data1=pd.read_csv('e_train.csv',sep=',',encoding='utf-8',low_memory=False) 

#count1 = data1.groupby('the concept of the topic').candidate.count()
#count1.to_csv('count1.csv')

t4 = data1[(data1['the concept of the topic']=='cannabis') & (data1['label']==1)]
t4cont = t4['candidate']
list_t4 = t4cont.tolist()

t5 = data1[(data1['the concept of the topic']=='cannabis') & (data1['label']==0)]
t5cont = t5['candidate']
list_t5 = t5cont.tolist()

t6 = data1[(data1['the concept of the topic']=='prostitution') & (data1['label']==0)]
t6cont = t6['candidate']
list_t6 = t6cont.tolist()

t7 = data1[(data1['the concept of the topic']=='prostitution') & (data1['label']==1)]
t7cont = t7['candidate']
list_t7 = t7cont.tolist()

t4feats=[(preprocess(str(list_t4[i])), 't4') for i in range(len(list_t4))]
t5feats=[(preprocess(str(list_t5[i])), 't5') for i in range(len(list_t5))]
t6feats=[(preprocess(str(list_t6[i])), 't6') for i in range(len(list_t6))]
t7feats=[(preprocess(str(list_t7[i])), 't7') for i in range(len(list_t7))]

t4cutoff = int(len(t4feats)*0.8)
t5cutoff = int(len(t5feats)*0.8)
t6cutoff = int(len(t6feats)*0.8)
t7cutoff = int(len(t7feats)*0.8)


# In[15]:


trainfeats1 =t4feats[:t4cutoff]+t5feats[:t5cutoff]+t6feats[:t6cutoff]+t7feats[:t7cutoff]
testfeats1 = t4feats[t4cutoff:]+t5feats[t5cutoff:]+t6feats[t6cutoff:]+t7feats[t7cutoff:]

print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats1)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats1))
classifier.show_most_informative_features()


# In[100]:


#data_xls = pd.read_excel('ce.xls', index_col=0)
#data_xls.to_csv('ces.csv', encoding='utf-8')

#import xlrd
#import csv
#import unicodecsv


#pip install xlrd

#workbook = xlrd.open_workbook('ce.xls', encoding_override="cp1251")
#table = workbook.sheet_by_index(0)
#with codecs.open('ce_s.csv', 'w', encoding='utf-8') as f:
#    write = csv.writer(f)
#    for row_num in range(table.nrows):
#        row_value = table.row_values(row_num)
#        write.writerow(row_value)

