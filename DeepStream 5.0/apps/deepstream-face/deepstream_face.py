#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys, time
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

import pyds
import cv2
import numpy as np

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

start = time.time()
prt = True

def osd_sink_pad_buffer_probe(pad,info,u_data):
    global start, prt
    frame_number=0

    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        now = time.time()
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            #frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            '''
            img = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            red = (0, 0, 255)
            location = (20, 50)
            font = cv2.FONT_ITALIC  # italic font
            cv2.putText(img, 'OpenCV Cooking', location, font, fontScale = 2, color = red, thickness = 3)
            #cv2.imshow('Hello', img)
            #cv2.waitKey(0)
            '''
            
            
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                '''
                print('class_id={}'.format(type(obj_meta.class_id)))
                print('confidence={}'.format(type(obj_meta.confidence)))
                print('detector_bbox_info={}'.format(type(obj_meta.detector_bbox_info)))
                print('obj_label={}'.format(type(obj_meta.obj_label)))
                print('object_id={}'.format(type(obj_meta.object_id)))
                print('rect_params={}'.format(type(obj_meta.rect_params)))
                '''
                #print('mask_params={}'.format(type(obj_meta.mask_params)))  #Not binded
                print('        rect_params bg_color alpha={}'.format(type(obj_meta.rect_params.bg_color)))
                print('    rect_params border_width={}'.format(type(obj_meta.rect_params.border_width)))
                print('    rect_params border_width={}'.format(obj_meta.rect_params.border_width))
                print('    rect_params  color_id={}'.format(type(obj_meta.rect_params.color_id)))
                print('    rect_params  color_id={}'.format(obj_meta.rect_params.color_id))
                print('    rect_params has_color_info={}'.format(type(obj_meta.rect_params.has_color_info)))


                '''
                if True:
                    print(' === obj_meta ===')
                    print('class_id={}'.format(obj_meta.class_id))
                    print('confidence={}'.format(obj_meta.confidence))
                    print('detector_bbox_info={}'.format(obj_meta.detector_bbox_info))
                    #print('mask_params={}'.format(obj_meta.mask_params))
                    print('obj_label={}'.format(obj_meta.obj_label))
                    print('object_id={}'.format(obj_meta.object_id))
                    print('rect_params={}'.format(obj_meta.rect_params))
                    print('        rect_params bg_color alpha={}'.format(obj_meta.rect_params.bg_color.alpha))
                    print('        rect_params bg_color blue={}'.format(obj_meta.rect_params.bg_color.blue))
                    print('        rect_params bg_color green={}'.format(obj_meta.rect_params.bg_color.green))
                    print('        rect_params bg_color red={}'.format(obj_meta.rect_params.bg_color.red))
                   
                    
                    print('        rect_params border_color alpha={}'.format(obj_meta.rect_params.border_color.alpha))
                    print('        rect_params border_color blue={}'.format(obj_meta.rect_params.border_color.blue))
                    print('        rect_params border_color green={}'.format(obj_meta.rect_params.border_color.green))
                    print('        rect_params border_color red={}'.format(obj_meta.rect_params.border_color.red))
                    print('    rect_params border_width={}'.format(obj_meta.rect_params.border_width))
                    print('    rect_params  color_id={}'.format(obj_meta.rect_params.color_id))
                    print('    rect_params has_bg_color={}'.format(obj_meta.rect_params.has_bg_color))
                    print('    rect_params has_color_info={}'.format(obj_meta.rect_params.has_color_info))
                    print('    rect_params height={}'.format(obj_meta.rect_params.height))
                    print('    rect_params left={}'.format(obj_meta.rect_params.left))
                    print('    rect_params top={}'.format(obj_meta.rect_params.top))
                    print('    rect_params width={}'.format(obj_meta.rect_params.width))
                    print('    rect_params reserved={}'.format(obj_meta.rect_params.reserved))


                    print('tracker_bbox_info={}'.format(obj_meta.tracker_bbox_info))
                    print('tracker_confidence={}'.format(obj_meta.tracker_confidence))
                '''
    
            except StopIteration:
                break

            obj_meta.rect_params.has_bg_color = 1
            obj_meta.rect_params.bg_color.set(0.0, 0.0, 1.0, 0.2) #It seems that only the alpha channel is working. RGB value is reflected.       
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(0.0, 1.0, 1.0, 0.0)    # It seems that only the alpha channel is not working. (red, green, blue , alpha)
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={} FPS={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON], (1 / (now - start)))

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        py_nvosd_text_params.font_params.font_size = 20
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(0.2, 0.2, 1.0, 1) # (red, green, blue , alpha)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.2, 0.2, 0.2, 0.3)
        # Using pyds.get_string() to get display_text as string
        if prt:
            print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        prt = False    
        start = now     
        
    return Gst.PadProbeReturn.OK    #DROP, HANDLED, OK, PASS, REMOVE 	


def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser \n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write(" Unable to create h264 parser \n")



    print("Creating MP4 \n")
    qtdemux = Gst.ElementFactory.make("qtdemux", "qt-demux")
    queue = Gst.ElementFactory.make("queue", "-queue")
    mp4parser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4video-parser")
    if not mp4parser:
        sys.stderr.write(" Unable to create mp4parser parser \n")


    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    print("Playing file %s " %args[1])
    source.set_property('location', args[1])
    #streammux.set_property('width', 1920)
    #streammux.set_property('height', 1080)
    streammux.set_property('width', 1024)
    streammux.set_property('height', 768)
    #streammux.set_property('width', 640)
    #streammux.set_property('height', 368)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "config_infer_primary_facedetectir.txt")

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(qtdemux)
    pipeline.add(queue)
    pipeline.add(mp4parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    if is_aarch64():
        pipeline.add(transform)

    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(h264parser)
    h264parser.link(decoder)


    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)
    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main(sys.argv))

