from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

count = len(results)

for result in results:
    for box in result.boxes.data:
        print("x1:%f y1:%f  x2[%f] y2[%f] Conf[%f] Label[%f]"%(box[0], box[1], box[2], box[3], box[4], box[5]))
