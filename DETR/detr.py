import torch as th
import torchvision
import torchvision.transforms as T
import requests, sys, time
from PIL import Image, ImageDraw, ImageFont
import argparse
import gc 

print('pytorch', th.__version__)
print('torchvision', torchvision.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="resnet50", help='network model -> resnet50 or resnet101 or resnet50_dc5 or  resnet50_panoptic')
parser.add_argument('--size', type=str, default="640X480", help='processing image size w X h  ex) -> 640X480')
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
    
model.eval()
model = model.cuda()
print('model[%s] load success'%args.model)

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


W = int(args.size.split('X')[0])
H = int(args.size.split('X')[1])
print('Image resize to W:%d  H:%d'%(W, H))

url = 'https://i.ytimg.com/vi/vrlX3cwr3ww/maxresdefault.jpg'
img = Image.open(requests.get(url, stream=True).raw).resize((W,H)).convert('RGB')

print('Image load success')
img_tens = transform(img).unsqueeze(0).cuda()

tfps = 0
count = 0
for i in range (5):
    fps_time  = time.perf_counter()
    th.cuda.empty_cache()
    gc.collect()
    with th.no_grad():
      output = model(img_tens)

    fps = 1.0 / (time.perf_counter() - fps_time)
    print("Net FPS: %f" % (fps))
    tfps += fps
    count += 1

    im2 = img.copy()
    drw = ImageDraw.Draw(im2)
    pred_logits=output['pred_logits'][0][:, :len(CLASSES)]
    pred_boxes=output['pred_boxes'][0]

    max_output = pred_logits.softmax(-1).max(-1)
    topk = max_output.values.topk(15)
    #print(topk)

    pred_logits = pred_logits[topk.indices]
    pred_boxes = pred_boxes[topk.indices]
    pred_logits.shape
    #print('topk.indices',topk.indices)


    for logits, box in zip(pred_logits, pred_boxes):
      m = th.nn.Softmax(dim=0)
      prob = m(logits)
      top3 = th.topk(logits, 3)
      print(' ===== print top3 values =====')
      print('top 1: Label[%-20s]  probability[%5.3f]'%(CLASSES[top3.indices[0]], prob[top3.indices[0]] * 100))
      print('top 2: Label[%-20s]  probability[%5.3f]'%(CLASSES[top3.indices[1]], prob[top3.indices[1]] * 100))
      print('top 3: Label[%-20s]  probability[%5.3f]'%(CLASSES[top3.indices[2]], prob[top3.indices[2]] * 100))
      
      cls = top3.indices[0]
      if cls >= len(CLASSES):
        continue
      label = CLASSES[cls]
      #print(label)
      box = box.cpu() * th.Tensor([W, H, W, H])
      x, y, w, h = box
      x0, x1 = x-w//2, x+w//2
      y0, y1 = y-h//2, y+h//2
      drw.rectangle([x0, y0, x1, y1], outline='red', width=5)
      drw.text((x, y), label, fill='white')
    
    fps = 1.0 / (time.perf_counter() - fps_time)
    print("FPS: %f" % (fps))
    output = None
    th.cuda.empty_cache()

print('Processing success')  
print('AVG FPS:%f'%(tfps / count))  
im2.save("./%s-%s.jpg"%(args.model, args.size))   
  