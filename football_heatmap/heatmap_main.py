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
from matplotlib.patches import Circle
import soccer_config
importlib.reload(soccer_config)
from soccer_config import SoccerPitchConfiguration
from keypoint_homography import Transform


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
        return position_array
    
    def generate_2dmapping(self,pitch,scaled_coord,all_frames,all_keypoints,position_array,img_pitch2d):
        """AI is creating summary for generate_2dmapping

        Args:
            pitch ([class object]): Class object that has properties like labels,pitch height, pitch width, vertices, labels etc
            scaled_coord (list): List of 32 scaled coordinates. Example:[(0, 0),(0, 145),(0, 258),....]
            all_frames (list): list of all the frames of  array
            all_keypoints (list): List of all the keypoints preent in the respective frame
            position_array (list): List of all interpolated ball position arrays
            img_pitch2d (array): Image array of the 2D pitch representation
        """
        labels=pitch.labels

        # Create a mapping from labels to vertices
        label_to_vertex = {label: vertex for label, vertex in zip(labels, scaled_coord)}
        frame_size = (1200, 700)

        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_file = f"videos/{video_name}_heatmap.avi"  # Output video file nam
        frame_rate =  self.fps
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the video
        video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)
        # Filter vertices based on the keypoints' classes
        for idx in range(len(all_frames)):
            filtered_vertices = []
            filtered_keypoints = []
            kpts=all_keypoints[idx]
            if not kpts == [{}] and len(kpts)>4:
                
                for kp in kpts:
                        label = kp['class']
                        if label in label_to_vertex:
                            filtered_vertices.append(label_to_vertex[label])
                            filtered_keypoints.append(kp['keypoints'])
                

                # Output the results
                #print("Filtered Vertices:", filtered_vertices)
                #print("Filtered Keypoints:", filtered_keypoints)

                final_vert_array=np.array(filtered_vertices)
                final_kpts_array=np.array(filtered_keypoints)
                transformer = Transform(
                    source=final_kpts_array.astype(np.float32),
                    target=final_vert_array.astype(np.float32)
                )
                pts_2d=position_array[idx].reshape(1, -1) 
                pts_2d
                final_img=transformer.transform_points(points=pts_2d)

                center = final_img[0] # Example center coordinates in (x, y)
                center=tuple(center.astype(int)) 

                # Define the radius of the circle
                radius = 9  # You can adjust this value as needed

                # Define the color of the circle (BGR format)
                color = (0, 0, 255)  # Red color

                # Define the thickness of the circle's border
                thickness = 5  # If you want a filled circle, use -1 for thickness

                # Draw the circle on the image
                temp_img=img_pitch2d.copy()
                cv2.circle(temp_img, center, radius, color, thickness)
            else:
                print("In else:")
            #plt.imshow(img)
            # Write the modified frame to the video
            video_writer.write(temp_img)
            #break
        video_writer.release()

    def generate_2Dheatmap(self,pitch,scaled_coord,all_frames,all_keypoints,position_array,img_pitch2d):
        labels=pitch.labels
        label_to_vertex = {label: vertex for label, vertex in zip(labels, scaled_coord)}
        img = img_pitch2d
        frame_height, frame_width, _ = img.shape

        # Initialize heatmap grid
        heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

        # Example ball positions - replace this with your data
        # ball_positions = [(x1, y1), (x2, y2), ...] collected earlier
        ball_positions = []  # Collect all ball positions for debugging

        # Collect ball positions (replace with your loop to update positions)
        for idx in range(len(all_frames)):
            kpts = all_keypoints[idx]
            if not kpts == [{}] and len(kpts) > 4:
                filtered_vertices = []
                filtered_keypoints = []

                for kp in kpts:
                    label = kp['class']
                    if label in label_to_vertex:
                        filtered_vertices.append(label_to_vertex[label])
                        filtered_keypoints.append(kp['keypoints'])

                final_vert_array = np.array(filtered_vertices)
                final_kpts_array = np.array(filtered_keypoints)
                transformer = Transform(
                    source=final_kpts_array.astype(np.float32),
                    target=final_vert_array.astype(np.float32)
                )
                pts_2d = position_array[idx].reshape(1, -1)
                final_img = transformer.transform_points(points=pts_2d)

                center = final_img[0]
                center = tuple(center.astype(int))

                # Update heatmap
                if 0 <= center[0] < frame_width and 0 <= center[1] < frame_height:
                    heatmap[center[1], center[0]] += 1  # Increment the cell
                    ball_positions.append(center)  # Collect positions

        # Apply Gaussian blur to the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (81, 81), 0)

        # Normalize the heatmap
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.power(heatmap, 0.3)
        # Convert heatmap to color
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Blend the heatmap with the original image
        alpha = 0.6  # Transparency factor
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

        # Display the result
        
        overlay_rgb=cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(overlay_rgb)
        plt.axis('off')
        plt.show()
        # Optionally, save the result
        cv2.imwrite("heatmap_overlay_18.png", overlay)


    def combine_video(self):

        video_name=os.path.splitext(os.path.basename(self.video_path))[0]
        # Paths to the original video and the heatmap video
        original_video_path =self.video_path
        heatmap_video_path = f"videos/{video_name}_heatmap.avi"
        output_combined_path = f'./videos/combined_output_{video_name}.avi'

        # Open both videos
        original_cap = cv2.VideoCapture(original_video_path)
        heatmap_cap = cv2.VideoCapture(heatmap_video_path)

        # Get video properties from the original video
        frame_width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(frame_width,frame_height)
        fps = int(original_cap.get(cv2.CAP_PROP_FPS))

        # Ensure both videos have the same number of frames
        original_frame_count = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        heatmap_frame_count = int(heatmap_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("original frame count:",original_frame_count)
        print("heatmap frame count:",heatmap_frame_count)
        #assert original_frame_count == heatmap_frame_count, "Videos must have the same number of frames!"

        # Define the codec and create the VideoWriter for the combined video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
        output_frame_width = frame_width * 2  # Side-by-side combination
        output_frame_height = frame_height  # No change in height
        out = cv2.VideoWriter(output_combined_path, fourcc, fps, (output_frame_width, output_frame_height))

        # Loop through the frames of both videos
        while original_cap.isOpened() and heatmap_cap.isOpened():
            ret1, original_frame = original_cap.read()
            ret2, heatmap_frame = heatmap_cap.read()

            if not ret1 or not ret2:
                break  # Stop if we reach the end of either video

            # Resize heatmap frame to match the original frame dimensions (if needed)
            heatmap_frame = cv2.resize(heatmap_frame, (frame_width, frame_height))

            # Combine the two frames side-by-side
            combined_frame = cv2.hconcat([original_frame, heatmap_frame])

            # Write the combined frame to the output video
            out.write(combined_frame)

        # Release all resources
        original_cap.release()
        heatmap_cap.release()
        out.release()

        print(f"Combined video saved as {output_combined_path}")


def perform_ball_detection(video_path,normal_detection=True):


    model_ball = get_model(
    model_id="football-ball-detection-rejhg/4", 
    api_key="fpgo7lotAA2MTZARBCtt")


    model_keypoints = get_model(
        model_id="football-field-detection-f07vi/14", 
        api_key="fpgo7lotAA2MTZARBCtt")
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


def generate_2d_field(pitch):
    
    # Define the image size (in pixels)
    img_width = 1200
    img_height = 700

   
    # Create a blank image (white background)
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Define scaling factors based on physical dimensions
    x_scale = img_width / pitch.length
    y_scale = img_height / pitch.width

    # Function to scale coordinates
    def scale_coordinates(x, y):
        return int(x * x_scale), int(y * y_scale)

    # List to store the scaled coordinates
    scaled_coord = []

    # Plot the vertices and edges
    for idx, (x, y) in enumerate(pitch.vertices):
        x_scaled, y_scaled = scale_coordinates(x, y)
        #print("The idx is: and its x y scaled is: ",idx,x,y)
        # Draw the vertex as a small circle
        cv2.circle(img, (x_scaled, y_scaled), 8, (0, 0, 0), -1)  # Black dot for the vertex
        # Annotate the point with label (if you need this step)
        cv2.putText(img, pitch.labels[idx], (x_scaled + 10, y_scaled + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save the scaled coordinates
        scaled_coord.append((x_scaled, y_scaled))

    # Plot the edges (lines connecting the vertices)
    for edge in pitch.edges:
        # Get the start and end vertices of the edge
        x1, y1 = pitch.vertices[edge[0] - 1]
        x2, y2 = pitch.vertices[edge[1] - 1]
        
        # Scale the coordinates to image space
        x1_scaled, y1_scaled = scale_coordinates(x1, y1)
        x2_scaled, y2_scaled = scale_coordinates(x2, y2)
        
        # Draw the line between the two points (green color)
        cv2.line(img, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)  # Green color for edges

    # Plot the center circle
    center_x, center_y = pitch.length / 2, pitch.width / 2  # The center of the field
    center_scaled_x, center_scaled_y = scale_coordinates(center_x, center_y)  # Scale to image coordinates
    centre_circle_radius_scaled = pitch.centre_circle_radius * x_scale  # Scale the radius to match the image

    # Draw the center circle
    cv2.circle(img, (center_scaled_x, center_scaled_y), int(centre_circle_radius_scaled), (0, 255, 0), 2)  # Green circle

    # Save the figure as an image file
    cv2.imwrite("soccer_pitch_with_circle_scaled_cv2.png", img)
    return img,scaled_coord





def main(video_path,normal_detection):
    file_exist,ball_pos_array_saved,keypoint_array_saved=check_saved_array(video_path)

    # Perform ball detection/ load the array if already detected
    if file_exist==True:
        print("Ball position and keypoint arrays exists so loading from saved array...")
        ball_pos_lst=ball_pos_array_saved
        keypoints_lst=keypoint_array_saved
        frames_lst=process_onlyvideo(video_path)
    else:
        print("Running ball detection...")
        ball_pos_lst,keypoints_lst,frames_lst=perform_ball_detection(video_path,normal_detection)
        save_datas(ball_pos_lst,keypoints_lst,video_path)


    #frame_to_save = frames_lst[100]  # Replace 0 with the index of the frame you want to save

    # Save the frame as an image
    #output_filename = 'saved_frame.jpg'  # You can choose any file name and format
    #cv2.imwrite(output_filename, frame_to_save)

    # Initialize Video_info class and extract infos
    # Perform ball interpolation and generate video 
    video_info = Video_info(video_path)
    interpolated_ballpos=video_info.ballpos_interpolation(ball_pos_lst,frames_lst)

    # generate a 2D Football field and scale the coordinate based on image size
     # Initialize soccer pitch configuration
    pitch = SoccerPitchConfiguration()

    pitch_2d,scaled_coordinates=generate_2d_field(pitch)
    print("Generating 2d mapping and writing video...")
    video_info.generate_2dmapping(pitch,scaled_coordinates,frames_lst,keypoints_lst,interpolated_ballpos,pitch_2d)

    print("Generating 2d heatmap...")
    video_info.generate_2Dheatmap(pitch,scaled_coordinates,frames_lst,keypoints_lst,interpolated_ballpos,pitch_2d)
    print("Combining videos...")
    video_info.combine_video()




    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a video with optional detection.')
    parser.add_argument('--video_path', metavar='path', required=True,
                        help='The path to the video file.')
    parser.add_argument('--normal_detection', action='store_true',
                        help='Enable normal detection. Defaults to False if not specified.')
    
    args = parser.parse_args()
    main(video_path=args.video_path, normal_detection=args.normal_detection)