import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import os
import cv2
import argparse
import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from threading import Thread
from utils import *
# %matplotlib inline

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'

# Make sure that caffe is on the python path:
import os
import sys
sys.path.append('./python')

import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'data/plate/labelmap_plate.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)


class LocalVideo:
    def __init__(self, src):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def isEmpty(self):
        return self.grabbed

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def parse_args():
    parser = argparse.ArgumentParser("video detector")
    parser.add_argument('-model',dest='model',help='path to model prototxt file',type=str)
    parser.add_argument('-weights',dest='weights', help='path to weight file',type=str)
    parser.add_argument('-video', dest='video', help='path to input video',type=str)
    parser.add_argument('-gpu', dest='gpu', help='specifiy using GPU or not', action='store_true')
    parser.add_argument('-threshold', dest='threshold', help='threshold to filter bbox with low confidence', type=float, default=0.3)
    return parser.parse_args()

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def get_color_map(num_classes):

    color_unit = int(255/num_classes)

    colors = []
    for i in range(num_classes):
        if colors == []:
            colors.append((color_unit, 0, 0))
        else:
            last = colors[-1]
            colors.append((last[2],last[0]+2*color_unit,last[1]+color_unit))

    return colors

if __name__ == '__main__':

    num_classes = 2

    args = parse_args()
    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()
    #Load the net in the test phase for inference, and configure input preprocessing.
    model_def = args.model
    model_weights = args.weights

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    #transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # DSOD detection

    # set net to batch size of 1
    image_resize_h = 300
    image_resize_w = 300
    batch_size = 1
    net.blobs['data'].reshape(batch_size, 3, image_resize_h, image_resize_w)

    # set colors
    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    colors = get_color_map(num_classes)

    #Load image.
    video_dir = args.video
    
    total_time = 1.0

    batch_list = []


    def single_report(val_gen, num):
        for i in range(num):
            test_inputs, test_targets, test_seq_len = val_gen.next_batch()
            test_feed = {inputs: test_inputs,
                         targets: test_targets,
                         seq_len: test_seq_len}
            dd = session.run(decoded[0], test_feed)
            detected_list = decode_sparse_tensor(dd)
            detected_string = ''.join(detected_list[0])
            print("Detected:", detected_string)
            return detected_string


    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        testi = './exp'
        # saver.restore(session, './model8.24best/LPRtf3.ckpt-25000')
        saver.restore(session, './model/LPRtf3.ckpt-15000')

        cap = cv2.VideoCapture(args.video)
        #cap = LocalVideo(args.video).start()
        # try to determine the total number of frames in the video file
        try:
          prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		           else cv2.CAP_PROP_FRAME_COUNT
          total = int(cap.get(prop))
          print("[INFO] {} total frames in video".format(total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
          print("[INFO] could not determine # of frames in video")
          print("[INFO] no approx. completion time can be provided")
          total = -1

        end = False
        fps = FPS().start()
        while True:
            ret, img = cap.read()
            #img = cap.read()
            if not ret:
               # drop the last batch
               break
            #img = imutils.resize(img, width=300)

            net.blobs['data'].data[...] = transformer.preprocess('data', img)

            # Forward pass.

            detections = net.forward()['detection_out']
            print('fps ',fps)
            # Parse the outputs.
            det_label = detections[0,0,:,1]
            det_conf = detections[0,0,:,2]
            det_xmin = detections[0,0,:,3]
            det_ymin = detections[0,0,:,4]
            det_xmax = detections[0,0,:,5]
            det_ymax = detections[0,0,:,6]

            # Get detections with confidence higher than 0.6.
            top_indices = [j for j, conf in enumerate(det_conf) if conf >= args.threshold]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_labels = get_labelname(labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            img = np.array(img, dtype=np.uint8)

            for k in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[k] * img.shape[1]))
                ymin = int(round(top_ymin[k] * img.shape[0]))
                xmax = int(round(top_xmax[k] * img.shape[1]))
                ymax = int(round(top_ymax[k] * img.shape[0]))
                score = top_conf[k]
                label = int(top_label_indices[k])
                label_name = top_labels[k]
                #display_txt = '%s: %.2f'%(label_name, score)
                p1 = (xmin, ymin)
                p2 = (xmax, ymax)
                org = (xmin, ymin - 10)
                color = colors[label]

                cv2.rectangle(img, p1, p2, color=color, thickness=2)
                print(xmin)
                print(xmax)
                print(ymax)
                print(ymin)
                cropped = img[ymin:ymax,xmin:xmax]
                cv2.imshow('crop',cropped)
                cv2.imwrite('./exp/11111111.jpg',cropped)
                test_gen = TextImageGenerator(img_dir=testi,
                                              label_file=None,
                                              batch_size=BATCH_SIZE,
                                              img_size=img_size,
                                              num_channels=num_channels,
                                              label_len=label_len)
                detected_plate = single_report(test_gen, 1)
                display_txt = '%s: %.2f' % (detected_plate, score)
                cv2.putText(img, display_txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #fps_str = 'fps: %.1f' % fps
            #fps_str ="FPS: {:.2f}".format(fps.fps())
            #cv2.putText(img, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 5)
            # show the output frame
            cv2.imshow("Frame", img)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # update the FPS counter
            fps.update()
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        cap.stop()

    #output_dir, name = os.path.split(args.video)
    #output_path = os.path.join(output_dir, 'out_'+name)

    #output_size = (output[0].shape[0], output[0].shape[1])
    #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #writer = cv2.VideoWriter(output_path, fourcc, 30, output_size)

    #for image in output:
    #    writer.write(image)


