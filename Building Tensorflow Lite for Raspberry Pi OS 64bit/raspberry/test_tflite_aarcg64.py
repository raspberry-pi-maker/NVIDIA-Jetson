import argparse
import sys
import time, json
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_support import metadata

parser = argparse.ArgumentParser(description='object detection')
parser.add_argument("--image", default="/home/pi/data/sample_image.jpg", help="test working directory where the image file exists")
parser.add_argument("--model", default="./efficientdet_lite4.tflite", help="model")
args = parser.parse_args()    


interpreter = Interpreter(model_path=args.model, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
#model requires these size
print('Inference Image Height:', height)
print('Inference Image Width:', width)
min_conf_threshold = 0.5

displayer = metadata.MetadataDisplayer.with_model_file(args.model)
model_metadata = json.loads(displayer.get_metadata_json())
# Load label list from metadata.
file_name = displayer.get_packed_associated_file_list()[0]
label_map_file = displayer.get_associated_file_buffer(file_name).decode()
label_list = list(filter(len, label_map_file.splitlines()))

image = cv2.imread(args.image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imH, imW, _ = image.shape 
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)


# Perform the actual detection by running the model with the image as input
interpreter.set_tensor(input_details[0]['index'],input_data)
interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects


for i in range(len(scores)):
    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

        # Get bounding box coordinates and draw box
        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        ymin = int(max(1,(boxes[i][0] * imH)))
        xmin = int(max(1,(boxes[i][1] * imW)))
        ymax = int(min(imH,(boxes[i][2] * imH)))
        xmax = int(min(imW,(boxes[i][3] * imW)))
        
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        object_name = label_list[int(classes[i])]
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        # Draw label

        
cv2.imwrite('./result.jpg', image)        
        