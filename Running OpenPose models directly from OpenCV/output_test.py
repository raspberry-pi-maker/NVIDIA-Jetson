import cv2
import time
import numpy as np
from random import randint

threshold = 0.2
#Body_25 model use 25 points
key_points = {
    0:  "Nose", 1:  "Neck", 2:  "RShoulder", 3:  "RElbow", 4:  "RWrist", 5:  "LShoulder", 6:  "LElbow",
    7:  "LWrist", 8:  "MidHip", 9:  "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee",
    14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe", 20: "LSmallToe",
    21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"
}

#Body_25 keypoint pairs 
POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],     #arm, shoulder line
              [1,8], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14],  #2 leg
              [11,24], [11,22], [22,23], [14,21],[14,19],[19,20],    #2 foot  
              [1,0], [0,15], [15,17], [0,16], [16,18], #face
              [2,17], [5,18]
               ]  
               
#Body_25 PAF information 46,47? 54,55?
mapIdx = [[40,41], [48,49], [42,43], [44,45], [50,51], [52,53],
          [26,27], [32,33], [28,29], [30,31], [34,35], [36,37], [38,39], #2 leg
          [76,77], [72,73], [74,75], [70,71], [66,67], [68,69], #2 foot
          [56,57], [58,59], [62,63], [60,61], [64,65], #face
          [46,47],[54,55]   #Rshoulder<->REar, Lshoulder<->LEar 
          ]    

alpha = 0.3

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]



protoFile = "/usr/local/src/openpose-1.7.0/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "/usr/local/src/openpose-1.7.0/models/pose/body_25/pose_iter_584000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

# img = cv2.imread('/usr/local/src/image/blackpink/blackpink.png')
img = cv2.imread('/usr/local/src/image/blackpink/blackpink.png')
frameWidth = img.shape[1]
frameHeight = img.shape[0]




inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)
Blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)
net.setInput(Blob)
output = net.forward()

nPoints = 25


for index in range(26, 78):
    probMap = output[0,index,:,:] * 255
    probMap = cv2.resize(probMap, (img.shape[1], img.shape[0]))
    probMap = np.asarray(probMap, np.uint8)
    probMap = cv2.cvtColor(probMap,cv2.COLOR_GRAY2BGR)
    dst = cv2.addWeighted(img, alpha, probMap, (1-alpha), 0)
    cv2.imwrite('/tmp/black_proMap_%d.jpg'%index, dst)


'''
for index in range(25):
    probMap = output[0,index,:,:]
    probMap = cv2.resize(probMap, (img.shape[1], img.shape[0]))
    keypoints = getKeypoints(probMap, threshold)

'''

                          