import cv2
from openpose import pyopenpose as op

params = dict()
params["model_folder"] = "/usr/local/src/openpose-1.7.0/models/"
params["net_resolution"] = "320x256"  #inference resolution

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


datum = op.Datum()
#imageToProcess = cv2.imread('/usr/local/src/image/blackpink/blackpink.png')
imageToProcess = cv2.imread('/usr/local/src/image/face.jpg')
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
newImage = datum.cvOutputData[:, :, :]
cv2.imwrite("/tmp/result.jpg", newImage)
