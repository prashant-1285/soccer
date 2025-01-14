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
import argparse



model_ball = get_model(
    model_id="football-ball-detection-rejhg/4", 
    api_key="fpgo7lotAA2MTZARBCtt"
)


model_keypoints = get_model(
    model_id="football-field-detection-f07vi/14", 
    api_key="fpgo7lotAA2MTZARBCtt"
)

class Video_info:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        # Check if the video was successfully opened
        if not self.cap.isOpened():
            print("Error: Unable to open video file.")
        else:
            # Get video properties for output writer
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.video_path=video_path

            
    def ballpos_interpolation(self,ball_pos,all_frames):
        print("Inside interpolation")
        yolo=False
        df_positions = pd.DataFrame(ball_pos, columns=["x", "y"])
        df_positions = df_positions.interpolate(method='linear').bfill()  # Interpolate missing values
        position_array=list(df_positions.values)

        # Define the codec and create VideoWriter object
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_path = f'./videos/{video_name}_track.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MP4V' for MP4 files
        out = cv2.VideoWriter(output_path, fourcc, self.fps , ( self.frame_width, self.frame_height))

        for current_positions,frame in zip(position_array,all_frames):
        
            

            # Get the current interpolated position for this frame
            x, y = current_positions
            
            if not np.isnan(x) and not np.isnan(y):
                
                x, y = int(x), int(y)
                # Draw interpolated circle
                cv2.circle(frame, (x, y), radius=10, color=(255, 0, 255), thickness=2)

            #Draw raw detections from YOLO
            if yolo:

                x1=int(x1)
                x2=int(x2)
                y1=int(y1)
                y2=int(y2)
                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                x,y,w,h=xyxy2xywh(x1,y1,x2,y2)

            out.write(frame)

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

def process_onlyvideo(video_path):

      # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    
    all_frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    return all_frames



def save_datas(ball_positions, all_keypoints, video_path):
    # Extract the video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Save the ball positions and keypoints arrays with video name
    ballpos_array = np.array(ball_positions)
    np.save(f"ballposition_array_{video_name}.npy", ballpos_array)
    
    all_keypoints_array = np.array(all_keypoints)
    np.save(f"keypoints_array_{video_name}.npy", all_keypoints_array)


def check_saved_array(video_path):
    exist=False
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if os.path.exists(f"ballposition_array_{video_name}.npy") and os.path.exists(f"keypoints_array_{video_name}.npy"):
        exist=True
  
        ball_pos_array=list(np.load(f"ballposition_array_{video_name}.npy",allow_pickle=True))
        keypoint_array=list(np.load(f"keypoints_array_{video_name}.npy",allow_pickle=True))
    else:
        exist=False
        ball_pos_array=None
        keypoint_array=None
    
    return exist,ball_pos_array,keypoint_array




def main(video_path,normal_detection):
    file_exist,ball_pos_array_saved,keypoint_array_saved=check_saved_array(video_path)

    if file_exist==True:
        print("Ball position and keypoint arrays exists so loading from saved array...")
        ball_pos_lst=ball_pos_array_saved
        keypoints_lst=keypoint_array_saved
        frames_lst=process_onlyvideo(video_path)
    else:
        print("Running ball detection...")
        ball_pos_lst,keypoints_lst,frames_lst=perform_ball_detection(video_path,normal_detection)
        save_datas(ball_pos_lst,keypoints_lst,video_path)

        # Initialize Video_info class and extract infos
    
    video_info = Video_info(video_path)
    video_info.ballpos_interpolation(ball_pos_lst,frames_lst)

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a video with optional detection.')
    parser.add_argument('--video_path', metavar='path', required=True,
                        help='The path to the video file.')
    parser.add_argument('--normal_detection', action='store_true',
                        help='Enable normal detection. Defaults to False if not specified.')
    
    args = parser.parse_args()
    main(video_path=args.video_path, normal_detection=args.normal_detection)