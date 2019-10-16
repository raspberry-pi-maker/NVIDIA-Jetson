import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import time
import cv2
import numpy as np
import gc 
import sys

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

IMG_SIZE = 480
THRESHOLD = 0.7
fps_time = 0

def process_frame(img):
  #out = None
  torch.cuda.empty_cache()
  gc.collect()
  fps_time  = time.perf_counter()
  img = cv2.resize(img, (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  #print('H:%d W:%d'%(img.shape[0], img.shape[1])) 
  trf = T.Compose([
      #T.ToPILImage(),
      T.ToTensor()
  ])

  input_tensor = trf(img)
  #input_tensor = torchvision.transforms.functional.to_tensor(img)
  #print(input_tensor.shape)
  input_img = [input_tensor.to(device)]
  out = model(input_img)[0]


  print(len(out['boxes']))
  for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
    #score = score.detach().numpy()
    score_np = score.cpu().detach().numpy()
    print(score_np)
    if score_np < THRESHOLD:
      continue

    #box = box.detach().numpy()
    box_np = box.to(torch.int16).cpu().numpy()
    #keypoints = keypoints.detach().numpy()[:, :2]
    keypoints_np = keypoints.to(torch.int16).cpu().numpy()[:, :2]

    cv2.rectangle(img, pt1=(int(box_np[0]), int(box_np[1])), pt2=(int(box_np[2]), int(box_np[3])), thickness=2, color=(0, 0, 255))

    for k in keypoints_np:
      cv2.circle(img, center=tuple(k.astype(int)), radius=2, color=(255, 0, 0), thickness=-1)

    cv2.polylines(img, pts=[keypoints_np[5:10:2].astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.polylines(img, pts=[keypoints_np[6:11:2].astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.polylines(img, pts=[keypoints_np[11:16:2].astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.polylines(img, pts=[keypoints_np[12:17:2].astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)

  fps = 1.0 / (time.perf_counter() - fps_time)
  new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cv2.putText(new_img , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  out_video.write(new_img)
  input_tensor.cpu()






if torch.cuda.is_available():
  device = torch.device('cuda')
  model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
  model = model.to(device)
  model.eval()
else:
  device = torch.device('cpu')
  model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

'''
device = torch.device('cpu')
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
'''


cap = cv2.VideoCapture('imgs/02.mp4')
ret, img = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter('imgs/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))

count = 1
 
while cap.isOpened():
  ret, img = cap.read()
  if ret == False:
      break
  process_frame(img)

  sys.stdout.flush ()
  print('Frame count[%d]'%count)
  count += 1
  
  #img = None
  #out = None
  #torch.cuda.empty_cache()

out_video.release()
cap.release()
