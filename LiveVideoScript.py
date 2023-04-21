#!/usr/bin/env python
# coding: utf-8

# In[8]:


"""
In this script, we will
"""

import torch 
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from sort import sort
from time import time
from collections import defaultdict
"""
Load the model with a certain confidence threshold (TBD)
"""
model = torch.hub.load('.','custom','best.pt',source='local')


# In[9]:


model.conf = .27 #Minimum .5 confidence threshold


# In[ ]:


"""
TODO:
1) Live Video
"""
frameRate = 0
time_limit = 100#10 Seconds
vid = cv2.VideoCapture(0)
ret,frame = vid.read()
image_frames = []

#Store tracking info per video
tracker=sort.Sort() 
ids = defaultdict(set)


# In[14]:


#.ttf file, font-size
myFont = ImageFont.truetype('ostrich-regular.ttf', 20)


# In[28]:


"""start_time = time()
ret,frame = vid.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
results = model(frame)
time_elaps = time()-start_time
fps = 1/time_elaps
print(time_elaps,fps)"""


# In[15]:


start_time = time()
#Set number of frames to look at for testing purposes
while(ret):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    results = model(frame)
    #Retrieving boundingbox data as dataframe 
    #Dataframe Format (xyxy attribute):
    #(xmin,ymin,xmax,ymax,confidence,label_value,label)
    data = results.pandas().xyxy[0]        
    trafficType=data["name"]
    data = data.iloc[:][:5].to_numpy()
    data = data[:,:5].astype('float64')


    #Updated with ids on the camera
    track_res = tracker.update(data)

    if(track_res.size!=0):
        for index,detection in enumerate(track_res[::-1]):
            #If the detection is new, add the ID to the
            #appropriate vehicle/pedestrian set
            #COUNT=LENGTH
            if(detection[-1] not in ids[trafficType[index]]):
                ids[trafficType[index]].add(detection[-1])

    #Image WITH BOX PREDICTIONS AND COUNT
    im = Image.fromarray(results.render(labels=False)[0])

    #Draw the count on the video frame
    im_draw = ImageDraw.Draw(im)
    draw_text=""
    for trafficType, id_set in ids.items():
        draw_text+=trafficType+" COUNT=" +str(len(id_set))+"\n"

    im_draw.multiline_text((0,
                      0), 
                 draw_text,
                 fill='white',font=myFont,
                 anchor = None, spacing = 0,
                 align="left",direction=None,
                 features=None,language=None,
                 stroke_width=1, stroke_fill="black")
    #Check if the images are being properly converted:
    image_frames.append(im)#Append for conversion to video
    time_elaps = time()-start_time
    if(time_elaps>time_limit):
        frameRate = len(image_frames)/(time_elaps)
        break
    ret, frame = vid.read()

#Save video as gif
if(len(image_frames)>0):
        #Save video as gif
        image_frames[0].save('runs/detect/videos/video1.gif',
                         save_all=True, optimize=False,append_images=image_frames[1:],loop=0)
else:
    print("No image frames to save!")

print("Frame rate: ",frameRate)


# In[ ]:




