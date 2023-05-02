#!/usr/bin/env python
# coding: utf-8

# In[67]:


# import required libraries
import torch 
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from sort import sort
from time import time
from collections import defaultdict
import threading
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2
from  datetime import datetime
import os.path as path
"""
Load the model with a certain confidence threshold (TBD)
"""
model = torch.hub.load('.','custom','best.pt',source='local')
model.conf = .4 #Minimum .5 confidence threshold


# In[68]:


def sendFrameToStream(frame, model_thread):
    while(model_thread.is_alive()):
        lock.acquire()
        writer.write(frame)
        lock.release()


# In[79]:


def model_image(frame, tracker, ids):
    """
    Takes an image frame and returns the traffic count in the frame 
    """
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
                lock.acquire()
                ids[trafficType[index]].add(detection[-1])
                lock.release()
    #Image WITH BOX PREDICTIONS AND COUNT
    im = Image.fromarray(results.render(labels=False)[0])

    #Draw the count on the video frame
    im_draw = ImageDraw.Draw(im)
    draw_text="Traffic Counts\n"

    for trafficType, id_set in ids.items():
        draw_text+=trafficType+"'s counted=" +str(len(id_set))+"\n"
        
    im_draw.multiline_text((0,
                      0), 
                 draw_text,
                 fill='white',font=myFont,
                 anchor = None, spacing = 0,
                 align="left",direction=None,
                 features=None,language=None,
                 stroke_width=1, stroke_fill="black")
    lock.acquire()
    frame = im
    lock.release()


# In[83]:


def updateCSV(trafficCounts, filename, currDatetime,thread):
    """
    trafficCounts: expected to be a dictionary of sets containing ids.
    We can count the vehicles by taking the length of each set. Each dictionary key
    matches our CSV columns
    
    filename: Assuming simple string of file name (no path)
    
    currDatetime: datetime.datetime object
    """
    while(thread.is_alive()):
        continue
    
    pd_data = None
    hours_passed = 0
    
    #If there is no file, create carTraffic file
    if(not path.isfile(filename)):
        trafficTypes = ["Articulated Truck", "Bicycle", "Bus", "Car", "Motorcycle",
                   "Motorized Vehicles", "Non-motorized Vehicles", "Pedestrian",
                   "Pickup Truck", "Single-Unit Truck", "Work Van"]
        hours = [str(x)+" Hour(s) ago" for x in reversed(range(1,13))]
        hours.append(datetime.now())
        pd_data= pd.DataFrame(data= 0,index = trafficTypes, columns = hours)
        
    else:
        pd_data= pd.read_csv(filename,index_col=0)

    #Accounting for when datetime is a string in csv
    if(type(pd_data.columns[-1])==str):
        hours_passed = currDatetime-datetime.strptime(pd_data.columns[-1], "%Y-%m-%d %H:%M:%S.%f")
        
    else:
        hours_passed = currDatetime- pd_data.columns[-1]
    
    #If more than an hour has passed shift all data to the left (columns)
    hours_passed = abs(hours_passed.total_seconds())/3600
    if(hours_passed > 0):
        pd_data = pd_data.shift(periods=-1,axis="columns",fill_value=0)
        pd_data.columns = [*pd_data.columns[:-1], datetime.now()]
        #Assuming data is correct, change elements of final column (Most recent hour)
        lock.acquire()
        for vehicleType in trafficCounts:
            pd_data.loc[vehicleType][-1] = len(trafficCounts[vehicleType])
        lock.release()
    else:    
        #Assuming data is correct, change elements of final column (Most recent hour)
        lock.acquire()
        for vehicleType in trafficCounts:
            pd_data.loc[vehicleType][-1] += len(trafficCounts[vehicleType])
        lock.release()

    pd_data.to_csv(filename)


# In[88]:


"""
Initializing values for tracking and video capture:
"""
#Store tracking info per video
tracker=sort.Sort() 
ids = defaultdict(set)
#.ttf file, font-size
myFont = ImageFont.truetype('ostrich-regular.ttf', 15)
# define and open video source
stream = CamGear(source="/dev/video0", logging=True).start()
# [WARNING] DO NOT INCLUDE IN GITHUB PUSH
YOUTUBE_STREAM_KEY = "1szm-jm2t-m45s-ygw9-1rbx"


# define required FFmpeg parameters for your writer
output_params = {
    "-clones": ["-f", "lavfi", "-i", "anullsrc"],
    "-vcodec": "libx264",
    "-preset": "medium",
    "-b:v": "4500k",
    "-bufsize": "512k",
    "-pix_fmt": "yuv420p",
    "-f": "flv",
}
# Define writer with defined parameters
writer = WriteGear(
    output="rtmp://a.rtmp.youtube.com/live2/{}".format(YOUTUBE_STREAM_KEY),
    logging=True,
    **output_params
)
lock = threading.Lock()


# loop over
while True:

    # read frames from stream
    frame = stream.read()
    # check for frame if Nonetype
    if frame is None:
        break
    
    thread1 = threading.Thread(target=model_image, kwargs={'frame':frame, 'tracker':tracker,'ids':ids})
    thread1.start()
    thread2 = threading.Thread(target=sendFrameToStream, kwargs={'frame':frame, 'model_thread':thread1})
    thread2.start()
    #Check if we can send data to csv
    curr_time = datetime.now()
    thread3 = threading.Thread(target=updateCSV, kwargs={'trafficCounts':ids, 
                                                         'filename':'smartTrafficData.csv',
                                                         'currDatetime':curr_time,
                                                        'thread':thread1})
    thread3.start()
    
    thread1.join()
    thread2.join()
    thread3.join()


# In[89]:


# safely close video stream
stream.stop()

# safely close writer
writer.close()


# In[ ]:




