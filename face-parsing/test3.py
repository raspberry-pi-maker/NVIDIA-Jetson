#!/usr/bin/python
# -*- encoding: utf-8 -*-
# som modification by spypiggy

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

def contour_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
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

    # comparison = vis_parsing_anno == vis_parsing_anno
    # print(comparison.all())

    print(vis_parsing_anno.shape)

    num_of_class = np.max(vis_parsing_anno)

    #We don't need to paint background, so index starts at 1
    for pi in range(1, num_of_class + 1):
        canvas = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]), dtype=np.uint8)
        index = np.where(vis_parsing_anno == pi)
        canvas[index[0], index[1]] = 255
        name = osp.join('./res/test_res', "%d_%s"%(pi, atts[pi]) + '.jpg')
        cv2.imwrite(name, canvas)
        contours, hierarchy = cv2.findContours(image=canvas, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image=vis_im, contours=contours, contourIdx=-1, color=part_colors[pi], thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(save_path[:-4] +'_contour.png', vis_im)


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
            out = ret.squeeze(0).cpu().numpy()
            print(out.shape)
            parsing = out.argmax(0)
            # print(parsing)
            # comparison = out[0] == parsing
            # equal_arrays = comparison.all()            
            # print(equal_arrays)
            # print(np.unique(parsing))
            contour_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Parcing')
    parser.add_argument("--image", default="./data", help="test working directory where the image file exists")
    parser.add_argument("--model", default="79999_iter.pth", help="faceparcing model")
    args = parser.parse_args()    

    CUDA_SUPPORT = torch.cuda.is_available()
    evaluate(dspth= args.image, cp=args.model)


