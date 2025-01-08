import numpy as np
import cv2
import os

def keypoint_prediction(model,frame,det_conf,keypt_conf):
    results = model.infer(frame, det_conf)[0]
    results_dict = dict(results)

    # Directly access predictions
    predictions = results_dict['predictions']
    #print("The predictions are: ",predictions)
    #print("the length of the predictions are: ",len(predictions))
    keypoint_detections=[]
    
    if not predictions:  # If predictions are empty
        keypoint_detections.append({})  # Append an empty dictionary
    else:
        for prediction in predictions:
            #print("Processing prediction with class name:", prediction.class_name)
        
            # Iterate through the keypoints
            for keypoint in prediction.keypoints:
                x, y = keypoint.x, keypoint.y
                class_name = keypoint.class_name
                confidence = keypoint.confidence
               
                if confidence>keypt_conf:
                    #print("The keeypoint confidence is : ",confidence)
                    # Append keypoint details in the desired format
                    keypoint_detections.append({
                        "keypoints": (x, y),
                        "class": class_name,
                        "conf": confidence
                    })
                # Print keypoint details
                #print(f"Keypoint class: {class_name}, x: {x}, y: {y}, confidence: {confidence}")
    return keypoint_detections

def draw_keypoints(all_keypoints,all_frames):
    drawn_frames=[]
    for frame, keypoints in zip(all_frames, all_keypoints):
        # Convert the frame from RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for keypoint_data in keypoints:
            # Extract keypoint details
            #print("keypoints adta is: ",keypoint_data)
            x, y = keypoint_data['keypoints']
            x=int(x)
            y=int(y)
            class_name = keypoint_data['class']
            confidence = keypoint_data['conf']

            # Draw the keypoint as a small circle
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            
            # Add the class name as text near the keypoint
            text = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, color=(0, 0, 255), thickness=1)
        drawn_frames.append(frame)
    return drawn_frames
        
    