from ultralytics import YOLO
import cv2
import time, sys
import torchvision
import torchvision.transforms as T

colors = [(255,0 , 0), (0,255,0), (0,0,255)]
font = cv2.FONT_HERSHEY_SIMPLEX   
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

def draw(img, boxes):
    index = 0
    for box in boxes.data:
        p1 =  (int(box[0].item()), int(box[1].item()))
        p2 =  (int(box[2].item()), int(box[3].item()))
        img = cv2.rectangle(img, p1, p2, colors[index % len(colors)], 3)
        text = label_map[int(box[5].item())] + " %4.2f"%(box[4].item()) 
        cv2.putText(img, text, (p1[0], p1[1] - 10), font, fontScale = 1, color = colors[index % len(colors)], thickness = 2)
        index += 1
    # cv2.imshow("draw", img)
    # cv2.waitKey(1)
    out_video.write(img)


# Load a model
model = YOLO("yolov8l.pt")  # load an official model
label_map = model.names

f = 0
net_total = 0.0
total = 0.0
reader = torchvision.io.VideoReader("./WUzgd7C1pWA.mp4", "video")
meta = reader.get_metadata()
fps = meta["video"]['fps'][0]
frame = next(reader)
img = T.ToPILImage()(frame["data"])
results = model(img)  # predict on an image

shape = frame['data'].shape     # pytorch image tensor shape is C H W
size = (frame['data'].shape[2], frame['data'].shape[1]) 
out_video = cv2.VideoWriter('./torch_result.mp4', fourcc, fps, size)


while frame:
    s = time.time()
    try:
        frame = next(reader)
    except StopIteration:    
        break
    img = T.ToPILImage()(frame["data"])
    results = model(img)  # predict on an image
    net_e = time.time()

    for result in results:
        draw(result.orig_img, result.boxes)
    e = time.time()
    net_total += (net_e - s)
    total += (e - s)
    f += 1

    
fps = f / total 
net_fps = f / net_total 

print("Total processed frames:%d"%f)
print("FPS:%4.2f"%fps)
print("Net FPS:%4.2f"%net_fps)
cv2.destroyAllWindows()
out_video.release()
