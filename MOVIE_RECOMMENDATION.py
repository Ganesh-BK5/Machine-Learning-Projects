#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies=pd.read_csv("Downloads\MM.csv")
credits=pd.read_csv("Downloads\MC.csv")


# In[3]:


print(movies)


# In[4]:


movies.head(2)


# In[5]:


movies.shape


# In[6]:


credits.head()


# In[7]:


movies=movies.merge(credits,on='title')


# In[8]:


movies.head()


# In[9]:


credits.cast[0]


# In[10]:


movies=movies[['movie_id','crew','cast','genres','title','overview','keywords']]


# In[11]:


movies.head()


# In[12]:


import ast


# In[13]:


def convert(text):
    L=[]
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies['genres']=movies['genres'].apply(convert)


# In[16]:


movies.head()


# In[17]:


movies['keywords']=movies['keywords'].apply(convert)


# In[18]:


movies.head()


# In[19]:


import ast
ast.literal_eval('[{"id":28,"name":"Action"},{"id":12,"name":"Adventure"},{"id":14,"name":"Fantasy"},{"id":878,"name":"Science Fiction"}]')


# In[20]:


def convert(text):
    L = []
    counter=0
    for i in ast.literal_eval(text):
        if counter<3:
            L.append(i['name'])
        counter+=1
    return L


# In[21]:


movies['cast']=movies['cast'].apply(convert)


# In[22]:


movies.head()


# In[23]:


movies['cast']=movies['cast'].apply(lambda x:x[0:3])


# In[24]:


def fetch_director(text):
    L=[]
    for i in ast.literal_eval(text):
        if i["job"]=='Director':
            L.append(i['name'])
    return L


# In[25]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[26]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[27]:


movies.sample(5)


# In[28]:


def collapse(L):
    L1=[]
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[29]:


movies['cast']=movies['cast'].apply(collapse)
movies['crew']=movies['crew'].apply(collapse)
movies['genres']=movies['genres'].apply(collapse)


# In[30]:


movies['keywords']=movies['keywords'].apply(collapse)


# In[31]:


movies.head()


# In[32]:


movies['overview']=movies['overview'].apply(lambda x: " ".join(x))


# In[33]:


movies.head()


# In[34]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[35]:


movies.head()


# In[36]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# movies.head()

# In[37]:


movies.head()


# In[38]:


new=movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[39]:


new


# In[40]:


new['tags']=new['tags'].apply(lambda x:" ".join(x))


# In[41]:


new.head()


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words="english")


# In[43]:


vector=cv.fit_transform(new["tags"]).toarray()


# In[44]:


vector.shape


# In[45]:


from sklearn.metrics.pairwise import cosine_similarity


# In[46]:


similarity=cosine_similarity(vector)


# In[47]:


similarity


# In[48]:


new[new['title']=='The Lego Movie'].index[0]


# In[49]:


def recommend(movie):
    index=new[new['title']==movie].index[0]
    distances=sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x:x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[51]:


recommend('Tanner Hall')


# In[ ]:





# In[ ]:





# In[ ]:




