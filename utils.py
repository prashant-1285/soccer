import numpy as np
import cv2
import os

def slice_frame(frame, slice_wh, overlap_wh):
    """Slices the frame into smaller patches with overlap."""
    height, width = frame.shape[:2]
    slice_width, slice_height = slice_wh
    overlap_width, overlap_height = overlap_wh
    patches = []

    y = 0
    while y < height:
        x = 0
        while x < width:
            patch = frame[y:min(y + slice_height, height), x:min(x + slice_width, width)]
            patches.append((x, y, patch))
            x += slice_width - overlap_width
        y += slice_height - overlap_height

    return patches




    # Extract bounding box information
def save_patches(patches,frame_number):
            # Iterate over patches to save each one
    # for i, row in enumerate(patches):
    #     for j, patch in enumerate(row):
    for i,patch in enumerate(patches):
            
            ptch=patch[2]
            
            # Save the patch as an image
            patch_filename = os.path.join("./patches", f"patch_{frame_number}_{i}.png")
            cv2.imwrite(patch_filename, ptch)
            print(f"Saved patch: {patch_filename}")


def compute_iou(box, boxes):
    """Compute Intersection over Union (IoU) between a box and a list of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection

    return intersection / (union + 1e-8)

def apply_nms(detections, iou_threshold):
    """Apply Non-Max Suppression to filter overlapping boxes."""
    if len(detections) == 0:
        return []

    detections = np.array(detections)  # Convert to NumPy for easier processing
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]

    # Sort by confidence score
    indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(indices) > 0:
        current_index = indices[0]
        selected_indices.append(current_index)
        current_box = detections[current_index]

        # Compute IoU for all remaining boxes
        iou = compute_iou(current_box, detections[indices[1:]])

        # Filter out boxes with IoU above the threshold
        indices = indices[1:][iou < iou_threshold]

    return detections[selected_indices]


def infer_on_patches(model, patches, frame_number,confidence,iou_threshold):
    """Runs inference on patches and combines results."""
    processed_patches = []
    all_detections = []
    detect_ball=False
    #print("Initial length of patches is: ",len(patches))
    for i,patch_info in enumerate(patches):
        x_offset, y_offset, patch = patch_info
        #print("x offset, y_offset ,patch number is: ",x_offset,y_offset,i)
        
        results = model.infer(patch, confidence)[0]
        results_dict = dict(results)

        # Directly access predictions
        predictions = results_dict['predictions']
      

        
        for prediction in predictions:
            conf = prediction.confidence
            #print("the confidence is: ",conf)
            if conf>= confidence:
                #print("Patch number with high confidence is: ",i)
                #print("Patch number with high confidence and it's offset is: ",x_offset, y_offset)
                #save_patches(patches,frame_number)
        
            
                detect_ball=True
                x_center = prediction.x
                #print("x center is:",x_center)
                y_center = prediction.y
                x_center = prediction.x
                y_center = prediction.y
                width = prediction.width
                height = prediction.height

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                global_x1 = x1 + x_offset
                global_y1 = y1 + y_offset
                global_x2 = x2 + x_offset
                global_y2 = y2 + y_offset
                #print("local x1,y1,x2,y2: ",x1,y1,x2,y2)
                #print("global x1,y1,x2,y2: ",global_x1,global_y1,global_x2,global_y2)

                all_detections.append([global_x1, global_y1, global_x2, global_y2, conf])
                #cv2.rectangle(patch, (x1, y1), (x2, y2), color, thickness)

        processed_patches.append((x_offset, y_offset, patch))
    
    # Apply Non-Max Suppression (NMS) to filter overlapping detections
    final_detections = apply_nms(all_detections, iou_threshold)
    # Draw final detections
   
    return final_detections,detect_ball

def reconstruct_frame(processed_patches, frame_shape):
    """Reconstructs the full frame from processed patches."""
    height, width = frame_shape[:2]
    reconstructed_frame = np.zeros((height, width, 3), dtype=np.uint8)

    for x_offset, y_offset, patch in processed_patches:
        # Get the dimensions of the patch
        patch_height, patch_width = patch.shape[:2]
        # Place the patch back into the reconstructed frame
        reconstructed_frame[y_offset:y_offset + patch_height, x_offset:x_offset + patch_width] = patch

    return reconstructed_frame