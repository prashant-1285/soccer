from ultralytics import YOLO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import skimage
from PIL import Image
from detection_utils import *
from keypoint_utils import *

import importlib
from sklearn.metrics import mean_squared_error
import json
import yaml
import time
import supervision as sv
from inference import get_model
import centroid_track
import detection_utils
import keypoint_utils
importlib.reload(centroid_track)
importlib.reload(detection_utils)
importlib.reload(keypoint_utils)



video_path="./videos/dbf_18.mp4"

model_ball = get_model(
    model_id="football-ball-detection-rejhg/4", 
    api_key="fpgo7lotAA2MTZARBCtt"
)


model_keypoints = get_model(
    model_id="football-field-detection-f07vi/14", 
    api_key="fpgo7lotAA2MTZARBCtt"
)
def perform_ball_detection(video_path,normal_detection=True):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    else:
    # Read the first frame
        pass

    # Get video properties for output writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))



    if not normal_detection:
        slice_dimensions = (640, 480)  # Width and height for slicing
        overlap_dimensions = (100, 100)  # Overlap between slices
        slice_wh=slice_dimensions
        overlap_wh=overlap_dimensions
    frame_number=0
    # Initialize the BallTracker
    tracker = centroid_track.BallTracker(buffer_size=5)


    ball_positions = []
    all_frames=[]
    frame_number=0
    # Read all the frames
    all_keypoints=[]
    det_conf=0.4
    keyp_conf=0.8
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

        if  normal_detection:
            #Run ball detection inference on a single FRAME
            detection_frame=normal_prediction(model_ball, frame, frame_number,0.3,0.4)
            #Run ball detection inference on patches
        else:
            patches = slice_frame(frame, slice_wh, overlap_wh)
            detection_frame,det_ball = infer_on_patches(model_ball, patches, frame_number,0.6,0.4)

        

        #print(" detection frame length is:  ",len(detection_frame))
        if len(detection_frame)>1:
            print("================Multiple BALL DETECTION==================")

        ## Run keypoint detection inference on single Frame

        kpoints = keypoint_utils.keypoint_prediction(model_keypoints,frame,det_conf,keyp_conf)
        if kpoints:
            all_keypoints.append(kpoints)
        else:
            print("Appending None KEYPOINTS")
            all_keypoints.append([{}])


        det=False
        detections_track = []

        if isinstance(detection_frame, np.ndarray) and detection_frame.size > 0:  # Check if it's a NumPy array with elements
            #print("inside the FIRST instance-----111")
            for dets in detection_frame:
                x1, y1, x2, y2, conf = dets
                x_centroid=(x1 + x2) // 2
                y_centroid=(y1 + y2) // 2
                detections_track.append([x_centroid,y_centroid ])  # Append the centroid
            detections_track = np.array(detections_track)
            tracked_ball = tracker.update(detections_track)

            ball_positions.append((tracked_ball[0], tracked_ball[1]))  # Store centroid
            det=True
            print("Detection is True")
        elif isinstance(detection_frame, list) and len(detection_frame) > 0:  # Check if it's a non-empty list
            for dets in detection_frame:
                x1, y1, x2, y2, conf = dets
                x_centroid=(x1 + x2) // 2
                y_centroid=(y1 + y2) // 2
                
                detections_track.append([x_centroid,y_centroid ]) 
            detections_track = np.array(detections_track)
            tracked_ball = tracker.update(detections_track)
            ball_positions.append((tracked_ball[0], tracked_ball[1]))  # Store centroid
            det=True
            print("Detection is True")
        else:
            #print("Inside none tracker-------")
            tracked_ball = tracker.update(None)  
            ball_positions.append((np.nan,np.nan))
            det=False
            print("Detection is False")
        print("The ball position is: ",ball_positions)
        print("The frame number is :",frame_number)
        frame_number+=1
        if frame_number==20:
            pass
            #break
    # Release the video capture object
    cap.release()

    return ball_positions,all_keypoints,all_frames