import numpy as np
import cv2
from ultralytics import YOLO
from byte_tracker import BYTETracker
# Load YOLO model
model = YOLO("yolov8n.pt")

# Input video source
src = "./home_foot.mp4"
cap = cv2.VideoCapture(src)

# Check if the video source is opened
if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_file = "./home_foot_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
output_resolution = (640, 480)  # Match the resized frame resolution
out = cv2.VideoWriter(output_file, fourcc, fps, output_resolution)


tracker = BYTETracker(fps)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or cannot fetch frame.")
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (640, 480))

    # Perform YOLO detection
    results = model(frame)

    detections = results[0]  # First result for the frame

    # Filter detections for 'person' class (class index 0 in COCO dataset)
    person_detections = detections.boxes[detections.boxes.cls == 0]
    print("person detection is: ",person_detections)
    
    detections_bytetrack = []
    # Get bounding box annotations (x1, y1, x2, y2)
    for box in person_detections:
        class_id='0'
        print("the box  value is : ",box)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        print("the box xyxy value is : ",box.xyxy[0])
        confidence = box.conf.cpu().numpy()
        bytetrack_bbox = [x1, y1, x2, y2,round(confidence, 2), int(class_id)]

        detections_bytetrack.append(bytetrack_bbox)

        # Draw bounding boxes on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"Person {confidence}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    detections_bytetrack_arry=np.array(detections_bytetrack)
    detection_update = tracker.update(detections_bytetrack_arry, fps)
    print(detection_update)

    # Write the annotated frame to the output video
    out.write(frame)

    # Press 'q' to exit the loop early (optional for debugging)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
print(f"Video saved to {output_file}")