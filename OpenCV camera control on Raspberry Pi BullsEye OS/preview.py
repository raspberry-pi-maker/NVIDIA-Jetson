import cv2
import sys

cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print('camera open Failed')
    sys.exit(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

while True:

    succes, img = cap.read()
    if succes == False:
        print('camera read Failed')
        sys.exit(0)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    cv2.imshow('Img',img)
cap.release()
cv2.destroyAllWindows()