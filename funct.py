#!/usr/bin/env python
# coding: utf-8

# # importing library

# In[102]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flair.models import TextClassifier
from flair.data import Sentence
from nltk.tokenize import sent_tokenize 
from statistics import mean 

import spacy
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import nltk
from flair.data import Sentence
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
nltk.download('punkt')
from autocorrect import Speller
import json

import warnings
warnings.filterwarnings("ignore")


# In[103]:


attributeRating=[
         {
            "name":"comfort",
            "rating":3.0
         },
         {
            "name":"quality",
            "rating":4.0
         },
         {
            "name":"running",
            "rating":5.0
         }]

comments=[
    "Value for money ",
    "Very poori quality ",
    "shoe is very comfortable.",
    "The quality of shoes are bad, the sole is not pasted properly and the material is of poor qualtiy. They are not comfortable and not long lasting. Very disappointed with the quality ",
    "This was not for jogging,Running,only walking . they are not comfortable ",
    "Package and products poor, but at this price it is value for money, thanks Amazon "
    
]


# In[105]:


def review(attributeRating,comments):
    attributeRating=attributeRating
    comments=comments
    attribute=[]
    def rating(sen):
        classifier = TextClassifier.load('en-sentiment')
        sc = Sentence(sen)
        classifier.predict(sc)
        x= float(str(sc.labels).split(" ")[1][1:-2])
        y=str(sc.labels).split(" ")[0][1:]
        #print(y)
        if ((x> 0.6) and (x<=1) and (y=="NEGATIVE")):
            return 1
        if ((x>- 0) and (x<0.4) and (y=="NEGATIVE") ):
            return 2
        if ((x>= 0.4 )and (x<0.6)):
            return 3
        if ((x>= 0) and (x<=0.4) and (y=="POSITIVE")):
            return 4
        if ((x> 0.6) and (x<=1) and (y=="POSITIVE")):
            return 5

    

    for i in range (len(attributeRating)):
        attribute.append(attributeRating[i]["name"])
    comments=map(str.lower, comments)
    comments=list(comments)
    result = pd.DataFrame({"comment":pd.Series(comments)}) 
    #comment = "The quality of shoes are bad, the sole is not pasted properly and the material is of poor qualtiy. They are not comfortable and not long lasting. Very disappointed with the quality "
    comsentences=[]
    for i in comments:
        for j in range(len(sent_tokenize(i))):
            comsentences.append(sent_tokenize(i)[j] )
    senti=pd.DataFrame({"sentence":pd.Series(comsentences)}) 
    senti['sentence']=senti['sentence'].str.lower()
    senti['sentence'] = senti['sentence'].str.replace('[^\w\s]','')
    check = Speller(lang='en')
    senti['sentence']=  senti['sentence'].apply(check)
    
    c=0
    for i in attribute :
        comfort=senti[senti['sentence'].str.contains((i))]
        comfort["rating"]=comfort["sentence"].apply(rating)
        attributeRating[c]["rating"]=comfort["rating"].mean(axis = 0, skipna = True)
        c=c+1
        
    return attributeRating
    
    

    
    


# In[106]:



rat=review(attributeRating,comments)
print(rat)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




