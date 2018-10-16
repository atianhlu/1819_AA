
# coding: utf-8

# In[88]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import numpy as np
import pandas as pd

data=pd.read_csv('c_train.csv',sep=',',encoding='utf-8',low_memory=False) 

#def word_feats(words):
#    return dict([(word, True) for word in words])

def readCSV2List(filePath):
    try:
        file=open(filePath,'r', encoding='UTF-8')
        context = file.read() 
        list_result=context.split('\n')
        length=len(list_result)
        for i in range(length):
            list_result[i]=list_result[i].split(',')
        return list_result
    finally:
        file.close();


# In[89]:


t1 = data[data['topic']=='Religion does more harm than good']
t1cont = t1['sentence']
t1cont.to_csv('t1.csv')

t2 = data[data['topic']=='Science is a major threat']
t2cont = t2['sentence']
t2cont.to_csv('t2.csv')

t3 = data[data['topic']=='Newspapers are outdated']
t3cont = t3['sentence']
t3cont.to_csv('t3.csv')

#t4 = data[data['the concept of the topic']=='cannabis']
#t4cont = t1['candidate']
#t4cont.to_csv('t4_cont.csv')

#t5 = data[data['the concept of the topic']=='wind power']
#t5cont = t1['candidate']
#t5cont.to_csv('t5_cont.csv')

#t6 = data[data['the concept of the topic']=='prostitution']
#t6cont = t1['candidate']
#t6cont.to_csv('t6_cont.csv')

data_t1=pd.read_csv('t1.csv',sep=',',encoding='utf-8',low_memory=False) 
data_t2=pd.read_csv('t2.csv',sep=',',encoding='utf-8',low_memory=False) 
data_t3=pd.read_csv('t3.csv',sep=',',encoding='utf-8',low_memory=False) 
#data_t4=pd.read_csv('t4_cont.csv',sep=',',encoding='utf-8',low_memory=False) 
#data_t5=pd.read_csv('t5_cont.csv',sep=',',encoding='utf-8',low_memory=False) 
#data_t6=pd.read_csv('t6_cont.csv',sep=',',encoding='utf-8',low_memory=False) 


def preprocess(s):
    return {word: True for word in s.lower().split()}


list_t1 = readCSV2List('t1.csv')
t1feats=[(preprocess(str(list_t1[i])), 't1') for i in range(len(list_t1))]

list_t2 = readCSV2List('t2.csv')
t2feats=[(preprocess(str(list_t2[i])), 't2') for i in range(len(list_t2))]

list_t3 = readCSV2List('t3.csv')
t3feats=[(preprocess(str(list_t3[i])), 't3') for i in range(len(list_t3))]

#list_t4 = readCSV2List('t4_cont.csv')
#t4feats=[(preprocess(str(list_t4[i])), 't4') for i in range(len(list_t4))]

#list_t5 = readCSV2List('t5_cont.csv')
#t5feats=[(preprocess(str(list_t5[i])), 't5') for i in range(len(list_t5))]

#list_t6 = readCSV2List('t6_cont.csv')
#t6feats=[(preprocess(str(list_t6[i])), 't6') for i in range(len(list_t6))]

t1cutoff = int(len(t1feats)*0.8)
t2cutoff = int(len(t2feats)*0.8)
t3cutoff = int(len(t3feats)*0.8)
#t4cutoff = int(len(t4feats)*0.8)
#t5cutoff = int(len(t5feats)*0.8)
#t6cutoff = int(len(t6feats)*0.8)

trainfeats = t1feats[:t1cutoff]+t2feats[:t2cutoff]+t3feats[:t3cutoff]#+t4feats[:t4cutoff]+t5feats[:t5cutoff]+t6feats[:t6cutoff]
testfeats = t1feats[t1cutoff:]+t2feats[t2cutoff:]+t3feats[t3cutoff:]#+t4feats[t4cutoff:]+t5feats[t5cutoff:]+t6feats[t6cutoff:]

print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))


# In[90]:


classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()


# In[92]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(trainfeats)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testfeats))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(trainfeats)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testfeats))


# In[40]:


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


# In[93]:


data1=pd.read_csv('e_train.csv',sep=',',encoding='utf-8',low_memory=False) 
df.head()


# In[94]:


count1 = data1.groupby('the concept of the topic').candidate.count()
count1.to_csv('count1.csv')


# In[95]:


#t1 = data[data['topic']=='Religion does more harm than good']
#t1cont = t1['sentence']
#t1cont.to_csv('t1.csv')

#t2 = data[data['topic']=='Science is a major threat']
#t2cont = t2['sentence']
#t2cont.to_csv('t2.csv')

#t3 = data[data['topic']=='Newspapers are outdated']
#t3cont = t3['sentence']
#t3cont.to_csv('t3.csv')

t4 = data1[(data1['the concept of the topic']=='cannabis') & (data1['label']==1)]
t4cont = t4['candidate']
t4cont.to_csv('t4_cont.csv')

t5 = data1[(data1['the concept of the topic']=='cannabis') & (data1['label']==0)]
t5cont = t5['candidate']
t5cont.to_csv('t5_cont.csv')

t6 = data1[(data1['the concept of the topic']=='prostitution') & (data1['label']==0)]
t6cont = t6['candidate']
t6cont.to_csv('t6_cont.csv')

t7 = data1[(data1['the concept of the topic']=='prostitution') & (data1['label']==1)]
t6cont = t7['candidate']
t6cont.to_csv('t7_cont.csv')


# In[96]:


#data_t1=pd.read_csv('t1.csv',sep=',',encoding='utf-8',low_memory=False) 
#data_t2=pd.read_csv('t2.csv',sep=',',encoding='utf-8',low_memory=False) 
#data_t3=pd.read_csv('t3.csv',sep=',',encoding='utf-8',low_memory=False) 
data_t4=pd.read_csv('t4_cont.csv',sep=',',encoding='utf-8',low_memory=False) 
data_t5=pd.read_csv('t5_cont.csv',sep=',',encoding='utf-8',low_memory=False) 
data_t6=pd.read_csv('t6_cont.csv',sep=',',encoding='utf-8',low_memory=False) 
data_t7=pd.read_csv('t6_cont.csv',sep=',',encoding='utf-8',low_memory=False) 


def preprocess(s):
    return {word: True for word in s.lower().split()}


#list_t1 = readCSV2List('t1.csv')
#t1feats=[(preprocess(str(list_t1[i])), 't1') for i in range(len(list_t1))]

#list_t2 = readCSV2List('t2.csv')
#t2feats=[(preprocess(str(list_t2[i])), 't2') for i in range(len(list_t2))]

#list_t3 = readCSV2List('t3.csv')
#t3feats=[(preprocess(str(list_t3[i])), 't3') for i in range(len(list_t3))]

list_t4 = readCSV2List('t4_cont.csv')
t4feats=[(preprocess(str(list_t4[i])), 't4') for i in range(len(list_t4))]

list_t5 = readCSV2List('t5_cont.csv')
t5feats=[(preprocess(str(list_t5[i])), 't5') for i in range(len(list_t5))]

list_t6 = readCSV2List('t6_cont.csv')
t6feats=[(preprocess(str(list_t6[i])), 't6') for i in range(len(list_t6))]

list_t7 = readCSV2List('t7_cont.csv')
t7feats=[(preprocess(str(list_t7[i])), 't7') for i in range(len(list_t7))]

#t1cutoff = int(len(t1feats)*0.8)
#t2cutoff = int(len(t2feats)*0.8)
#t3cutoff = int(len(t3feats)*0.8)
t4cutoff = int(len(t4feats)*0.8)
t5cutoff = int(len(t5feats)*0.8)
t6cutoff = int(len(t6feats)*0.8)
t7cutoff = int(len(t7feats)*0.8)

#trainfeats = t1feats[:t1cutoff]+t2feats[:t2cutoff]+t3feats[:t3cutoff]#+
trainfeats =t4feats[:t4cutoff]+t5feats[:t5cutoff]+t6feats[:t6cutoff]+t7feats[:t7cutoff]
#testfeats = t1feats[t1cutoff:]+t2feats[t2cutoff:]+t3feats[t3cutoff:]#+
testfeats = t4feats[t4cutoff:]+t5feats[t5cutoff:]+t6feats[t6cutoff:]+t7feats[t7cutoff:]

print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))


# In[97]:


classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()


# In[100]:


data_xls = pd.read_excel('ce.xls', index_col=0)
data_xls.to_csv('ces.csv', encoding='utf-8')


# In[104]:


import xlrd
import csv
import unicodecsv


# In[108]:


pip install xlrd


# In[106]:


workbook = xlrd.open_workbook('ce.xls', encoding_override="cp1251")
table = workbook.sheet_by_index(0)
with codecs.open('ce_s.csv', 'w', encoding='utf-8') as f:
    write = csv.writer(f)
    for row_num in range(table.nrows):
        row_value = table.row_values(row_num)
        write.writerow(row_value)

