from ultralytics import YOLO
import cv2

colors = [(255,0 , 0), (0,255,0), (0,0,255)]
font = cv2.FONT_HERSHEY_SIMPLEX   
def draw(img, boxes):
    index = 0
    for box in boxes.data:
        p1 =  (int(box[0].item()), int(box[1].item()))
        p2 =  (int(box[2].item()), int(box[3].item()))
        img = cv2.rectangle(img, p1, p2, colors[index % len(colors)], 3)
        text = label_map[int(box[5].item())] + " %4.2f"%(box[4].item()) 
        cv2.putText(img, text, (p1[0], p1[1] - 10), font, fontScale = 1, color = colors[index % len(colors)], thickness = 2)
        index += 1
    cv2.imwrite("./result.jpg", img)    
    # cv2.imshow("draw", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Load a model
model = YOLO("yolov8n.pt")  # load an official model
label_map = model.names
# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

count = len(results)

for result in results:
    draw(result.orig_img, result.boxes)
