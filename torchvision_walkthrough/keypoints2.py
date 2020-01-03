import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import argparse
import sys, time

IMG_SIZE = 480
THRESHOLD = 0.95


parser = argparse.ArgumentParser(description="Keypoint detection. - Pytorch")
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

print('cuda:', args.cuda)
if True == torch.cuda.is_available():
    print('pytorch:%s GPU support'% torch.__version__)
else:
    print('pytorch:%s GPU Not support ==> Error:Jetson should support cuda'% torch.__version__)
    sys.exit()
print('torchvision', torchvision.__version__)

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
if(args.cuda):
    model = model.cuda()

#img = Image.open('imgs/07.jpg')
img = Image.open('imgs/apink1.jpg')
img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width)))

plt.figure(figsize=(16, 16))
plt.imshow(img)


trf = T.Compose([
        T.ToTensor()
        ])

input_img = trf(img)
print(input_img.shape)
if(args.cuda):
    input_img = input_img.cuda()

fps_time  = time.perf_counter()

out = model([input_img])[0]
print(out.keys())


codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]

fig, ax = plt.subplots(1, figsize=(16, 16))
ax.imshow(img)

for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
    if(args.cuda):
        score = score.cpu().detach().numpy()
    else:        
        score = score.detach().numpy()

    if score < THRESHOLD:
        continue

    if(args.cuda):
        box = box.to(torch.int16).cpu().numpy()
        keypoints = keypoints.to(torch.int16).cpu().numpy()[:, :2]
    else:
        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy()[:, :2]

    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    # 17 keypoints
    for k in keypoints:
        circle = patches.Circle((k[0], k[1]), radius=2, facecolor='r')
        ax.add_patch(circle)
    
    # draw path
    # left arm
    path = Path(keypoints[5:10:2], codes)
    line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    ax.add_patch(line)
    
    # right arm
    path = Path(keypoints[6:11:2], codes)
    line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    ax.add_patch(line)
    
    # left leg
    path = Path(keypoints[11:16:2], codes)
    line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    ax.add_patch(line)
    
    # right leg
    path = Path(keypoints[12:17:2], codes)
    line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    ax.add_patch(line)

plt.savefig('result.jpg')
fps = 1.0 / (time.perf_counter() - fps_time)

if(args.cuda):
    print('FPS(cuda support):%f'%(fps))
else:    
    print('FPS(cuda not support):%f'%(fps))

