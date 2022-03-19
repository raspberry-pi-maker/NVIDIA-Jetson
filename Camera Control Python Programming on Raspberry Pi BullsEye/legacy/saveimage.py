from picamera import PiCamera,  Color
from time import sleep
f_res = None

def set_resolution(ver, index):
    global f_res, camera
    res = {"v1":[(2592,1944),(1920, 1280),(1296, 972), (1296, 730),(640, 480)],
           "v2":[(3280,2464),(2592,1944),(1920, 1280),(1296, 972), (1296, 730),(640, 480)]
           }
    f_res = res[ver][index]
    print("final resolution", f_res)
    camera.resolution = f_res
    

camera = PiCamera()
set_resolution("v2", 0)
# camera.annotate_background = Color('blue')
# camera.annotate_foreground = Color('yellow')
# camera.annotate_text = " Hello world "
# camera.brightness = 70
camera.start_preview(fullscreen=False,window=(0,0,1280,960))

sleep(5)
camera.capture('/home/pi/src/legacy/picamera_cap.jpg')
camera.stop_preview()
