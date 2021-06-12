#https://dev.classmethod.jp/articles/mediapipe-extract-data-from-multi-hand-tracking/
import cv2
import sys, os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
file = '/usr/local/src/example/body.jpg'
file = '/usr/local/src/example/face.png'
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
os.chdir('/usr/local/src/mediapipe')
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:

    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
        print('No face_landmarks')
        sys.exit(0)
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
    cv2.imwrite('/usr/local/src/example/facemesh_image.png', annotated_image)