import argparse
import time
import face_recognition

parser = argparse.ArgumentParser(description='face match run')
parser.add_argument('--known', type=str, default='./img/known/Bill Gates.jpg')
parser.add_argument('--unknown', type=str, default='./img/unknown/bill-gates-4.jpg')
args = parser.parse_args()

s_time = time.perf_counter()
image_of_known = face_recognition.load_image_file(args.known)
known_face_encoding = face_recognition.face_encodings(image_of_known)[0]

unknown_image = face_recognition.load_image_file(args.unknown)
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding, tolerance=0.7)
e_time = time.perf_counter()

if results[0]:
    print('[%s] [%s] are the same person'%(args.known, args.unknown))
else:
    print('[%s] [%s] are NOT the same person'%(args.known, args.unknown))

print('Elapsed Time:%5.2f seconds'%(e_time - s_time))