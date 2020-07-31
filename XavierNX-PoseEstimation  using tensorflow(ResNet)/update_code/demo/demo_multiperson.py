import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__) + "/../")

#from scipy.misc import imread, imsave
from PIL import Image, ImageDraw, ImageFont
import time

from util.config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
import argparse

#from multiperson.visualize import PersonDraw, visualize_detections
#import matplotlib.pyplot as plt

'''
Total 17 points in COCO 
'''
def validate_coco_pose(pose):
    err = 0
    for p in pose:
        if p[0] < 0.1 or p[1] < 0.1 :
            err += 1
    if err > 8 : 
        return False
    return True
    
def draw_coco_points(image, persons):
    fontname = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fnt = ImageFont.truetype(fontname, 15)
    draw = ImageDraw.Draw(image)
    radius = 3
    clr = (0,255,0)
    draw_person = 0
    thickness = 3
    for j in range(len(persons)):
        pose = persons[j]
        if False == validate_coco_pose(pose):
            continue
        draw_person += 1    
        #if j < 11:
        #    continue
        for i in range(len(pose)):
            p = pose[i]
            cx = p[0]
            cy = p[1]
            if cx < 0.1 and cy < 0.1 :
                continue
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline = clr, width=3)
            draw.text((cx + 10, cy), "%d"%i, font=fnt, fill=(255,255,255))
    
        #draw nose -> REye (0-> 2)
        if all(pose[0]) and all(pose[2]):
            draw.line([tuple(pose[0]), tuple(pose[2])],width = thickness, fill=(219,0,219))
        #draw nose -> LEye (0-> 1)
        if all(pose[0]) and all(pose[1]):
            draw.line([tuple(pose[0]), tuple(pose[1])],width = thickness, fill=(219,0,219))

        #draw LEye ->LEar(1-> 3)
        if all(pose[1]) and all(pose[3]):
            draw.line([tuple(pose[1]), tuple(pose[3])],width = thickness, fill=(219,0,219))
        #draw REye ->REar(2-> 4)
        if all(pose[2]) and all(pose[4]):
            draw.line([tuple(pose[2]), tuple(pose[4])],width = thickness, fill=(219,0,219))

        #draw RShoulder ->RHip(6-> 12)
        if all(pose[6]) and all(pose[12]):
            draw.line([tuple(pose[6]), tuple(pose[12])],width = thickness, fill=(153,0,51))
        #draw LShoulder ->LHip(5-> 11)
        if all(pose[5]) and all(pose[11]):
            draw.line([tuple(pose[5]), tuple(pose[11])],width = thickness, fill=(153,0,51))

        #draw RShoulder -> LShoulder (6-> 5)
        if all(pose[6]) and all(pose[5]):
            draw.line([tuple(pose[6]), tuple(pose[5])],width = thickness, fill=(255,102,51))

        #draw RShoulder -> RElbow(6-> 8)
        if all(pose[6]) and all(pose[8]):
            draw.line([tuple(pose[6]), tuple(pose[8])],width = thickness, fill=(255,255,51))
        #draw RElbow -> RWrist (8 ->10)
        if all(pose[8]) and all(pose[10]):
            draw.line([tuple(pose[8]), tuple(pose[10])],width = thickness, fill=(255,255,51))

        #draw LShoulder -> LElbow (5-> 7 )
        if all(pose[5]) and all(pose[7]):
            draw.line([tuple(pose[5]), tuple(pose[7])],width = thickness, fill=(51,255,51))
        #draw LElbow -> LWrist (7 ->9)
        if all(pose[7]) and all(pose[9]):
            draw.line([tuple(pose[7]), tuple(pose[9])],width = thickness, fill=(51,255,51))

        #draw RHip -> RKnee (12 ->14)
        if all(pose[12]) and all(pose[14]):
            draw.line([tuple(pose[12]), tuple(pose[14])],width = thickness, fill=(51,102,51))
        #draw RKnee -> RFoot (14 ->16)
        if all(pose[14]) and all(pose[16]):
            draw.line([tuple(pose[14]), tuple(pose[16])],width = thickness, fill=(51,102,51))

        #draw LHip -> LKnee(11 ->13)
        if all(pose[11]) and all(pose[13]):
            draw.line([tuple(pose[11]), tuple(pose[13])],width = thickness, fill=(51,51,204))
        #draw LKnee -> LFoot (13 ->15)
        if all(pose[13]) and all(pose[15]):
            draw.line([tuple(pose[13]), tuple(pose[15])],width = thickness, fill=(51,51,204))
    
    return image, draw_person


parser = argparse.ArgumentParser(description="Tensorflow Pose Estimation Example")
parser.add_argument("--image", type=str, default = "demo/image_multi.png", help="image file name")
args = parser.parse_args()

cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

#draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
file_name = args.image
#image = imread(file_name, mode='RGB')
image = Image.open(file_name).convert('RGB')
image_batch = data_to_input(image)
start = time.time()
# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

detections = extract_detections(cfg, scmap, locref, pairwise_diff)
unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)
end = time.time()
print(person_conf_multi)
print('===== Net FPS :%f ====='%( 1 / (end - start)))
image, draw_person = draw_coco_points(image, person_conf_multi)
image.save('./multi_coco_result[%d].png'%draw_person)

end = time.time()
print('===== FPS :%f ====='%( 1 / (end - start))) 

'''
img = np.copy(image)
visim_multi = img.copy()
fig = plt.imshow(visim_multi)
draw_multi.draw(visim_multi, dataset, person_conf_multi)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.show()
visualize.waitforbuttonpress()
'''