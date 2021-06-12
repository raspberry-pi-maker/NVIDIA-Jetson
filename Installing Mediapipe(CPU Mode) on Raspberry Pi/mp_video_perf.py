#https://dev.classmethod.jp/articles/mediapipe-extract-data-from-multi-hand-tracking/
import cv2
import sys, time, os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def get_face_mesh(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
        return image
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        #print(' face_landmarks:', face_landmarks)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
        #print('%d facemesh_landmarks'%len(face_landmarks.landmark))
    return annotated_image
    
os.chdir('/usr/local/src/mediapipe')    
font = cv2.FONT_HERSHEY_SIMPLEX    
cap = cv2.VideoCapture(0)
if (cap.isOpened() == False): 
  print("Unable to read camera feed")    
  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while cap.isOpened():
    s = time.time()
    ret, img = cap.read()  
    if ret == False:
        print('WebCAM Read Error')    
        sys.exit(0)
        
    annotated = get_face_mesh(img)
    e = time.time()
    fps = 1 / (e - s)
    cv2.putText(annotated, 'FPS:%5.2f'%(fps), (10,50), font, fontScale = 1,  color = (0,255,0), thickness = 1)
    cv2.imshow('webcam', annotated)
    key = cv2.waitKey(1)
    if key == 27:   #ESC
        break

cap.release()
