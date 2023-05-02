#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
In this script, we will
"""

import torch 
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from sort import sort
from time import time
from collections import defaultdict
import threading
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2
"""
Load the model with a certain confidence threshold (TBD)
"""
model = torch.hub.load('.','custom','best.pt',source='local')
model.conf = .4 #Minimum .5 confidence threshold


# In[2]:


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

# [WARNING] Change your YouTube-Live Stream Key here:
YOUTUBE_STREAM_KEY = "1szm-jm2t-m45s-ygw9-1rbx"


# In[3]:


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
def sendFrameToStream(frame):
    lock.acquire()
    writer.write(frame)
    lock.release()
    return

# In[4]:


def model_image(frame):
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
                ids[trafficType[index]].add(detection[-1])

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
    return

# In[5]:
start_time = time()

# loop over
while True:
    if(time() - start_time >= 60):
        #safely close video stream
        stream.stop()

        # safely close writer
        writer.close()

        print("Finished Live Video Stream")
        break
    # read frames from stream
    frame = stream.read()
    # check for frame if Nonetype
    if frame is None:
        break
    
    model_image(frame)
    sendFrameToStream(frame)
    
    thread1 = threading.Thread(target=model_image, kwargs={'frame':frame})
    thread1.start()
    thread2 = threading.Thread(target=sendFrameToStream, kwargs={'frame':frame})
    thread2.start()
    #Run while thread1 is modeling
    while(thread1.is_alive()):
        if(not thread2.is_alive()):
            thread2 = threading.Thread(target=sendFrameToStream, kwargs={'frame':frame})
            thread2.start()
    thread2.join()
    thread1.join()
