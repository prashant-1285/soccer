import numpy as np
from collections import deque

# Initialize the BallTracker class
class BallTracker:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections):
        #print("the detection inside the balltracker: ",detections)
        
        if detections is None or len(detections) == 0:
            return None

        # Calculate the centroid of previous detections
        #print("The length of buffer is: ",len(self.buffer))
        #print("the buffer values are: ",self.buffer)
        if len(self.buffer) > 0:
            centroid = np.mean(np.concatenate(self.buffer), axis=0)
            #print("the centroid is: ",centroid)
        else:
            centroid = np.mean(detections, axis=0)

        # Find the detection closest to the centroid
        distances = np.linalg.norm(detections - centroid, axis=1)
        print("the distances are: ",distances)

        index = np.argmin(distances)
        print("the index are: ",index)
        self.buffer.append(detections[index].reshape(1, -1))

        return detections[index]