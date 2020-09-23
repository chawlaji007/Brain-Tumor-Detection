# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:35:35 2020

@author: hp
"""
from keras.models import load_model
from werkzeug.utils import secure_filename
from keras.applications.imagenet_utils import decode_predictions
classifier=load_model('brain_tumor.h5')

from flask import Flask,request
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
#file1=request.files.get('file')
img1 = image.load_img('Healthy.jpeg',target_size=(64,64))
img = image.img_to_array(img1)
img = img/255



# create a batch of size 1 [N,H,W,C]
Pred_img = np.expand_dims(img, axis=0)
prediction = classifier.predict(Pred_img, batch_size=None,steps=1) #gives all class prob.


if(prediction[:,:]>0.65):
    value ='Yes :%1.2f'%(prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    value ='No :%1.2f'%(1.0-prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
plt.imshow(img1)
plt.show()