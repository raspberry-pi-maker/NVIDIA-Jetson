from picamera import PiCamera, PiCameraValueError
from time import sleep
f_res = None

def set_resolution(ver, index):
    global f_res, camera
    res = {"v1":[(2592,1944),(1920, 1280),(1296, 972), (1296, 730),(1024, 768),(800, 600),(640, 480)],
           "v2":[(3280,2464),(2592,1944),(1920, 1280),(1296, 972), (1296, 730),(1024, 768),(800, 600),(640, 480)]
           }
    f_res = res[ver][index]
    print("final resolution", f_res)
    camera.resolution = f_res
camera = PiCamera()
set_resolution("v2", 3)
camera.start_preview(fullscreen=False,window=(0,0,1280,960))
try:
    camera.start_recording('/home/pi/src/legacy/picamera.h264')
    sleep(5)
    camera.stop_recording()
except PiCameraValueError as err:
    print("Picamera Err:", err)
    print("Please use another resolution")

camera.stop_preview()
    