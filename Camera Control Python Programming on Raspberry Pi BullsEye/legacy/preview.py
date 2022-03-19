from picamera import PiCamera
from time import sleep

f_res = None
def set_resolution(ver, index):
    global f_res, camera
    res = {"v1":[(2592,1944),(1920, 1280),(1296, 972), (1296, 730),(640, 480)],
           "v2":[(2592,1944),(1920, 1280),(1296, 972), (1296, 730),(640, 480)]
           }
    f_res = res[ver][index]
    print("final resolution", f_res)
    camera.resolution = f_res
    
camera = PiCamera()
set_resolution("v1", 4)
camera.start_preview(fullscreen=False,window=(0,0,f_res[0],f_res[1]))
sleep(5)
camera.stop_preview()