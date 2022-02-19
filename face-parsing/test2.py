#!/usr/bin/python
# -*- encoding: utf-8 -*-
# some modification by spypiggy

import argparse
from model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

atts = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

def analyze(out):
    np.set_printoptions(precision=1)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf) 
    out = out.squeeze(0).cpu().numpy()
    print('model out shape:', out.shape )
    parsing = np.zeros((512, 512))
    for x in range(512):
        for y in range(512):
            pixel = out[:,x,y]
            val =  pixel.argmax(0)
            parsing[x, y] = val
    length = out.shape[0]
    parsing = out.argmax(0)
    print('parcing shape:', parsing.shape )

    for i in range(length):
        img = out[i, :,:]
        img16 = img.astype(np.int16)
        img8 = img.astype(np.uint8)
        # print("index [%d] arg min:%d max:%d "%(i, np.min(m), np.max(m)))
        # print("index [%d] 16 bit min:%d max:%d  8 bit min:%d max %d"%(i, np.min(img16), np.max(img16), np.min(img8), np.max(img8)))

    print("parsing min:%d max:%d  "%( np.min(parsing), np.max(parsing)))

    # print('squeezed shape:', out.shape )
    # parsing = out.argmax(0)
    # print('argmax shape:', parsing.shape )
    # skinfloat = out[1, :,:]
    # print(skinfloat)
    # skin = skinfloat.astype(np.uint8)
    # print(skin)
    # cv2.imshow("skin",skin)
    # cv2.waitKey(0)


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts (BGR)
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [0, 255, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [128, 128, 128], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    #parsing_anno original dtype:int64 so convert to uint8(0 ~ 255)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # print(vis_parsing_anno.shape)
    # vis_parsing_anno2 = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    # comparison = vis_parsing_anno == vis_parsing_anno
    # print(comparison.all())

    print(vis_parsing_anno.shape)

    num_of_class = np.max(vis_parsing_anno)

    #We don't need to paint background, so index starts at 1
    for pi in range(1, num_of_class + 1):
        canvas = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
        canvas[index[0], index[1], :] = part_colors[pi]
        name = osp.join('./res/test_res', "%d_%s"%(pi, atts[pi]) + '.jpg')
        cv2.imwrite(name, canvas)
        


    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path[:-4] +'.jpg', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im



def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    save_pth = osp.join('./res/cp', cp)

    if CUDA_SUPPORT:
        net.cuda()
        net.load_state_dict(torch.load(save_pth))
    else:
        device = torch.device('cpu')
        net.load_state_dict(torch.load(save_pth, map_location=device))

    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            if CUDA_SUPPORT:
                img = img.cuda()

            ret = net(img)[0]
            analyze(ret)

            out = ret.squeeze(0).cpu().numpy()
            print(out.shape)
            parsing = out.argmax(0)
            # print(parsing)
            # comparison = out[0] == parsing
            # equal_arrays = comparison.all()            
            # print(equal_arrays)
            # print(np.unique(parsing))
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Parcing')
    parser.add_argument("--image", default="./data", help="test working directory where the image file exists")
    parser.add_argument("--model", default="79999_iter.pth", help="faceparcing model")
    args = parser.parse_args()    

    CUDA_SUPPORT = torch.cuda.is_available()
    evaluate(dspth= args.image, cp=args.model)


