import subprocess, os, sys, time

exec = '/usr/local/src/openpose-1.7.0/build/examples/openpose/openpose.bin'
working_dir = '/usr/local/src/openpose-1.7.0/'
image_dir_info = "/usr/local/src/image"
write_images_info = "/usr/local/src/output"
write_json_info = "/usr/local/src/output"
net_resolution_info = "320x224"

params = []
params.append(exec)
params.append("--image_dir")
params.append(image_dir_info)
params.append("--write_images")
params.append(write_images_info)
params.append("--write_json")
params.append(write_json_info)
params.append("--net_resolution")
params.append(net_resolution_info)
#no display will speed up the processing time.
params.append("--display")
params.append("0")

#chdir is necessary. If not set, searching model path may fail
os.chdir(working_dir)
s = time.time()
process = subprocess.Popen(params, stdout=subprocess.PIPE)
output, err = process.communicate()
exit_code = process.wait()
e = time.time()
output_str = output.decode('utf-8')
print("Python Logging popen exit code :%d"%exit_code)
print("Python Logging popen return :%s"%output_str)
print("Python Logging Total Processing time:%6.2f"%(e - s))

