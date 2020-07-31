import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

#from scipy.misc import imread
from PIL import Image, ImageDraw, ImageFont
import time
from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input


def draw_mpii_points(image, pose):
    fontname = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fnt = ImageFont.truetype(fontname, 15)
    draw = ImageDraw.Draw(image)
    radius = 3
    clr = (0,255,0)
    for i in range(len(pose)):
        p = pose[i]
        cx = p[0]
        cy = p[1]
        accuracy = p[2]
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline = clr, width=3)
        draw.text((cx + 10, cy), "%d"%i, font=fnt, fill=(255,255,255))
    
    #draw 
    draw.line([tuple(pose[0][:2]), tuple(pose[1][:2]), tuple(pose[2][:2]), tuple(pose[3][:2]), tuple(pose[4][:2]), tuple(pose[5][:2]) ],width = 2, fill=(255,255,0))
    draw.line([tuple(pose[6][:2]), tuple(pose[7][:2]), tuple(pose[8][:2]), tuple(pose[9][:2]), tuple(pose[10][:2]), tuple(pose[11][:2])  ],width = 2, fill=(255,255,0))
    draw.line([tuple(pose[12][:2]), tuple(pose[13][:2])],width = 2, fill=(255,255,0))
    draw.line([tuple(pose[12][:2]), tuple(pose[8][:2]), tuple(pose[9][:2]), tuple(pose[12][:2])],width = 2, fill=(255,255,0))
    draw.line([tuple(pose[8][:2]), tuple(pose[2][:2])],width = 2, fill=(255,255,0))
    draw.line([tuple(pose[9][:2]), tuple(pose[3][:2])],width = 2, fill=(255,255,0))
        
    image.save('./mpii_result.png')


cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
file_name = "demo/image.png"
#image = imread(file_name, mode='RGB')
image = Image.open(file_name).convert('RGB')

image_batch = data_to_input(image)

start = time.time()
# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)
# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
end = time.time()
print('===== Net FPS :%f ====='%( 1 / (end - start))) 
print(pose)
draw_mpii_points(image, pose)
end = time.time()
print('===== FPS :%f ====='%( 1 / (end - start))) 




# Visualise
#visualize.show_heatmaps(cfg, image, scmap, pose)
#visualize.waitforbuttonpress()
