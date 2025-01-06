import yolov5

# load pretrained model
model = yolov5.load('yolov5s.pt')


  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img = 'frames.jpg'

# perform inference
results = model(img)


print("results",results)
# parse results
predictions = results.pred[0]
print("predictions",predictions)
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]



# save results into "results/" folder
#results.save(save_dir='results/')