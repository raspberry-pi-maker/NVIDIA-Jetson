import argparse
import time
from PIL import Image, ImageDraw 
import face_recognition

s_time = time.time()
parser = argparse.ArgumentParser(description='face match run')
parser.add_argument('--image', type=str, default='./img/groups/team2.jpg')
args = parser.parse_args()

image = face_recognition.load_image_file(args.image)
face_locations = face_recognition.face_locations(image)

pil_image = Image.fromarray(image)
img = ImageDraw.Draw(pil_image) 

# Array of coords of each face
# print(face_locations)
for face_location in face_locations:
    top, right, bottom, left = face_location
    img.rectangle([(left, top),(right, bottom)], fill=None, outline='green')

print(f'There are {len(face_locations)} people in this image')
pil_image.save('findface_result.jpg')
e_time = time.time()
elapsed = e_time - s_time
print('Elapsed Time:%f seconds'%(elapsed))
