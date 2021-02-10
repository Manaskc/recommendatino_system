#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[153]:


def get_titleindex(index):
    return df[df.index==index]["title"].values[0]


# In[154]:


def get_indextile(title):
    return df[df.title==title]["index"].values[0]


# In[155]:


df=pd.read_csv("movie_dataset.csv")


# In[156]:


df.head()


# In[157]:


df.columns


# In[158]:


colu = ['keywords','genres','cast','director']


# In[159]:


for i in colu:
    df[i]=df[i].fillna('')


# In[160]:


def combine_fea(row):
    try:
        return row['keywords']+" "+row['genres']+" "+row['cast']+" "+row['director']
    except:
        print (row)


# In[161]:


df["combine_fea"]=df.apply(combine_fea,axis=1)


# In[162]:


df.combine_fea.head()


# In[163]:


countvec= CountVectorizer()
count_matrix=countvec.fit_transform(df["combine_fea"])
cosin=cosine_similarity(count_matrix)


# In[164]:


sd=count_matrix.toarray()


# In[165]:


print(sd)


# In[166]:


sd.shape


# In[167]:


sd1=print (cosin)


# In[168]:


movie="Spectre"


# In[169]:


movie_index= get_indextile(movie)
inside_index=list(enumerate(cosin[movie_index]))


# In[171]:


sort_movie=sorted(inside_index,key=lambda x:x[1],reverse=True)


# In[179]:


i=0
for ir in sort_movie:
    sd3=get_titleindex(ir[0])
    print (sd3)
    i=i+1
    if i>10:
        break


# In[ ]:




