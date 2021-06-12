import torch as th
import torchvision
import torchvision.transforms as T
import requests, sys, time, os
from PIL import Image, ImageDraw, ImageFont
import argparse
import gc 

print('pytorch', th.__version__)
print('torchvision', torchvision.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="resnet50", help='network model -> resnet50 or resnet101 or resnet50_dc5 or  resnet50_panoptic')
parser.add_argument("--threshold", type=float, default=0.7, help="minimum detection threshold to use")
args = parser.parse_args()

'''
#if you want to view supported models, use these codes.
name = th.hub.list('facebookresearch/detr');
print(name)
'''
if args.model == 'resnet50':
    model = th.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
elif args.model == 'resnet50_dc5':
    model = th.hub.load('facebookresearch/detr', 'detr_resnet50_dc5', pretrained=True)
elif args.model == 'resnet50_dc5_panoptic':
    model = th.hub.load('facebookresearch/detr', 'detr_resnet50_dc5_panoptic', pretrained=True)
elif args.model == 'resnet50_panoptic':
    model = th.hub.load('facebookresearch/detr', 'detr_resnet50_panoptic', pretrained=True)

elif args.model == 'resnet101':
    model = th.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
elif args.model == 'resnet101_dc5':
    model = th.hub.load('facebookresearch/detr', 'detr_resnet101_dc5', pretrained=True)
elif args.model == 'resnet101_dc5_panoptic':
    model = th.hub.load('facebookresearch/detr', 'detr_resnet101_dc5_panoptic', pretrained=True)
elif args.model == 'resnet101_panoptic':
    model = th.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True)
    
else:    
    print('Unknown network name[%s]'%(args.model))
    sys.exit(0)

t1 = time.time()    
model.eval()
model = model.cuda()
print('model[%s] load success'%args.model)
t2 = time.time()
print("======== Network Load time:%f"%(t2 - t1))


transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
COLORS = [(0, 45, 74, 127), (85, 32, 98, 127), (93, 69, 12, 127),
          (49, 18, 55, 127), (46, 67, 18, 127), (30, 74, 93, 127)]
          
src_path = '/usr/local/src/test_images'
dest_path = '/usr/local/src/result'

fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 16)
fnt2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)


for root, dirs, files in os.walk(src_path):
    for fname in files:
        gc.collect()
        full_fname = os.path.join(root, fname)
        img = Image.open(full_fname).convert('RGB')
        W, H = img.size
        t1 = time.time()
        img_tens = transform(img).unsqueeze(0).cuda()

        fps_time  = time.perf_counter()
        with th.no_grad():
          output = model(img_tens)

        elapsed = time.time() - t1
        fps = 1.0 / elapsed
        print("FPS:%f"%(fps))


        im2 = img.copy()
        drw = ImageDraw.Draw(im2, 'RGBA')
        pred_logits=output['pred_logits'][0]
        pred_boxes=output['pred_boxes'][0]

        color_index = 0
        for logits, box in zip(pred_logits, pred_boxes):
            m = th.nn.Softmax(dim=0)
            prob = m(logits)
            top3 = th.topk(logits, 3)
            if top3.indices[0] >= len(CLASSES) or prob[top3.indices[0]] < args.threshold:
                continue
              
            print(' ===== print top3 values =====')
            print('top3', top3)

            print('top 1: Label[%-20s]  probability[%5.3f]'%(CLASSES[top3.indices[0]], prob[top3.indices[0]] * 100))

            if top3.indices[1] < len(CLASSES) :
                print('top 2: Label[%-20s]  probability[%5.3f]'%(CLASSES[top3.indices[1]], prob[top3.indices[1]] * 100))

            if top3.indices[2] < len(CLASSES) :
                print('top 3: Label[%-20s]  probability[%5.3f]'%(CLASSES[top3.indices[2]], prob[top3.indices[2]] * 100))
            
            
            cls = top3.indices[0]
            label = '%s-%4.2f'%(CLASSES[cls], prob[cls] * 100 )
            print(label)
            box = box.cpu() * th.Tensor([W, H, W, H])
            x, y, w, h = box
            x0, x1 = x-w//2, x+w//2
            y0, y1 = y-h//2, y+h//2
            color = COLORS[color_index % len(COLORS)]
            color_index += 1            
            drw.rectangle([x0, y0, x1, y1], fill = color, width=5)
            drw.text((x, y), label, font=fnt,fill='white')
            
        output = None
        th.cuda.empty_cache()
        
        drw.text((5, 5), 'FPS-%4.2f'%(fps), font=fnt2,fill='green')
        s = os.path.splitext(fname)[0]
        out_name = os.path.join(dest_path, s + '_detr_%s_%4.2f.jpg'%(args.model, fps))
        im2.save(out_name)   

