import cv2

def extract_video_segment(video_path, output_path, start_time, end_time):
    """
    Extracts a segment of the video between start_time and end_time in seconds.
    
    Args:
    - video_path (str): Path to the input video file.
    - output_path (str): Path to save the extracted segment.
    - start_time (int): Start time in seconds.
    - end_time (int): End time in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate the start and end frames
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate the frame range
    if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
        print("Error: Invalid start or end time.")
        return
    
    # Set up the VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Read frames and write the segment
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video segment saved to {output_path}")

# Example usage
video_path = "videos/dbf.mp4"
output_path = "dbf_test.mp4"
start_time = 2  # in seconds
end_time = 10   # in seconds

extract_video_segment(video_path, output_path, start_time, end_time)
