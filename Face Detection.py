#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
import seaborn as sns


# In[26]:


face_detector=cv2.CascadeClassifier("Downloads//haarcascade_frontalface_default.xml")


# In[27]:


def detect(imgp):
    img=cv2.imread(imgp)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    faces=face_detector.detectMultiScale(gray,1.1,4)
    for(x,y,w,h)in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
    plt.imshow(img)
    plt.show()

    
    


# In[28]:


detect('Downloads//people2.jpg')


# In[30]:


detect('Downloads//11.jpg')

