# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image, ImageDraw
from pose_engine import PoseEngine
import argparse

parser = argparse.ArgumentParser(description='PoseNet')
parser.add_argument('--model', type=str, default='mobilenet')
args = parser.parse_args()

os.system('wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/'
          'Hindu_marriage_ceremony_offering.jpg/'
          '640px-Hindu_marriage_ceremony_offering.jpg -O couple.jpg')
pil_image = Image.open('couple.jpg')

if(args.model == 'mobilenet'):
    pil_image.resize((641, 481), Image.NEAREST)
    engine = PoseEngine('models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
else:
    pil_image.resize((640, 480), Image.NEAREST)
    engine = PoseEngine('models/resnet/posenet_resnet_50_640_480_16_quant_edgetpu_decoder.tflite')
    
poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
print('Inference time: %.fms' % inference_time)

output = pil_image.copy()
draw = ImageDraw.Draw(output)
for pose in poses:
    if pose.score < 0.4: continue
    print('\nPose Score: ', pose.score)
    for label, keypoint in pose.keypoints.items():
        print(' %-20s x=%-4d y=%-4d score=%.1f' %
              (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
        p1 = (keypoint.yx[1], keypoint.yx[0])
        p2 = (keypoint.yx[1] + 5, keypoint.yx[0] + 5)
        draw.ellipse([p1, p2], fill=(0,255,0,255))
        draw.text((keypoint.yx[1] + 10,keypoint.yx[0] - 10), label,  fill=(0,255,0,128))
        
output.save('./couple_' + args.model + '.jpg')        