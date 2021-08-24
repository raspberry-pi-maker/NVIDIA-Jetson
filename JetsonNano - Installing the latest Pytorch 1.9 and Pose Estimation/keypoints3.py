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


parser = argparse.ArgumentParser(description="Keypoint detection. - Pytorch")
parser.add_argument('--image', type=str, default="./imgs/03.jpg", help='inference image')
parser.add_argument('--accuracy', type=float, default=0.9, help='accuracy. default=0.6')
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

if True == torch.cuda.is_available():
    print('pytorch:%s GPU support'% torch.__version__)
else:
    print('pytorch:%s GPU Not support ==> Error:Jetson should support cuda'% torch.__version__)
print('torchvision', torchvision.__version__)

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
if(args.cuda):
    model = model.cuda()

img = Image.open(args.image)
#img = Image.open('imgs/apink1.jpg')
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


#The first result is time consuming. After the second, check the processing time with the result.
#model([input_img])
fps_time  = time.perf_counter()
out = model([input_img])[0]
print(out.keys())
t_human = 0
r_human = 0



codes = [
    Path.MOVETO,
    #Path.LINETO,
    Path.LINETO
]

fig, ax = plt.subplots(1, figsize=(16, 16))
ax.imshow(img)
t_human = 0
r_human = 0
for box, score, keypoints, kscores  in zip(out['boxes'], out['scores'], out['keypoints'], out['keypoints_scores'] ):
    if(args.cuda):
        score = score.cpu().detach().numpy()
        kscores = kscores.cpu().detach().numpy()    
        box = box.to(torch.int16).cpu().numpy()
        keypoints = keypoints.to(torch.int16).cpu().numpy()[:, :2]
    else:        
        score = score.detach().numpy()
        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy()[:, :2]
        kscores = kscores.detach().numpy()    

    t_human += 1
    if score < args.accuracy:
        continue
    r_human += 1

    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    # 17 keypoints
    #for k in keypoints:
    for x in range(len(keypoints)):
        k = keypoints[x]
        if kscores[x] > 0:
            if x == 5:
                circle = patches.Circle((k[0], k[1]), radius=4, facecolor='r')
            else:
                circle = patches.Circle((k[0], k[1]), radius=2, facecolor='r')
            ax.add_patch(circle)
    
    # draw path
    # left arm
    if kscores[5] > 0 and kscores[7] > 0:
        path = Path(keypoints[5:8:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
    if kscores[7] > 0 and kscores[9] > 0:
        path = Path(keypoints[7:10:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
    
    # right arm
    if kscores[6] > 0 and kscores[8] > 0:
        path = Path(keypoints[6:9:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
    if kscores[8] > 0 and kscores[10] > 0:
        path = Path(keypoints[8:11:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)


    # left leg
    if kscores[11] > 0 and kscores[13] > 0:
        path = Path(keypoints[11:14:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
    if kscores[13] > 0  and kscores[15] > 0:
        path = Path(keypoints[13:16:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
    

    # right leg
    if kscores[12] > 0 and kscores[14] > 0:
        path = Path(keypoints[12:15:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
    if kscores[14] > 0 and kscores[16] > 0:
        path = Path(keypoints[14:17:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)

plt.savefig('result.jpg')
fps = 1.0 / (time.perf_counter() - fps_time)
print('total human:%d  real human:%d'%(t_human, r_human))

if(args.cuda):
    print('FPS(cuda support):%f'%(fps))
else:    
    print('FPS(cuda not support):%f'%(fps))
