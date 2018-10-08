
# coding: utf-8

# In[9]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import numpy as np
import pandas as pd

data=pd.read_csv('e_train.csv',sep=',',encoding='utf-8',low_memory=False) 

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

neg = data[data['label']==0]
negcont = neg['candidate']
negcont.to_csv('neg_cont.csv')

pos = data[data['label']==1]
poscont = pos['candidate']
poscont.to_csv('pos_cont.csv')

data_neg=pd.read_csv('neg_cont.csv',sep=',',encoding='utf-8',low_memory=False) 
data_pos=pd.read_csv('pos_cont.csv',sep=',',encoding='utf-8',low_memory=False) 

def preprocess(s):
    return {word: True for word in s.lower().split()}

list_neg = readCSV2List('neg_cont.csv')
negfeats=[(preprocess(str(list_neg[i])), 'neg') for i in range(len(list_neg))]
list_pos = readCSV2List('pos_cont.csv')
posfeats=[(preprocess(str(list_pos[i])), 'pos') for i in range(len(list_pos))]

negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)
 
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()


# In[10]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(trainfeats)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testfeats))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(trainfeats)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testfeats))


# In[11]:


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

