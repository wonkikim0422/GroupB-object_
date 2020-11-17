#! /usr/bin/env python3
from __future__ import division

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import roslib
import torchvision
from torchvision import datasets, models, transforms

from std_msgs.msg import String
from core_msgs.msg import yolomsg
from sensor_msgs.msg import CompressedImage

import cv2
import rospy
import roslib
import rospkg

from PIL import Image

from models import *
from utils.utils import *
from utils.datasets import *

import time
import datetime
import argparse

from torch.utils.data import DataLoader
from torch.autograd import Variable

def pad_to_square(img, pad_value):
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

        return img, pad


data_T = transforms.Compose([
    # transforms.Resize(512),
    # transforms.CenterCrop(224),
    transforms.ToTensor()])


class yolo_detection:
    def __init__(self):

        self.start_flag = 0

        self.pub = rospy.Publisher('/detected_objects', yolomsg, queue_size=1)

        parser = argparse.ArgumentParser()
        parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
        parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
        parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
        parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
        parser.add_argument("--conf_thres", type=float, default=0.95, help="object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
        parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
        self.opt = parser.parse_args()

        self.image_sub = rospy.Subscriber("/camera2/rgb/image_raw/compressed", CompressedImage, self.callback, queue_size=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnt = 0

        self.model =  Darknet(self.opt.model_def, img_size=self.opt.img_size).to(self.device)  # Set up model

        if self.opt.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.opt.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.opt.weights_path))

        self.model.eval()  # Set in evaluation mode

        self.start_flag = 1

    def callback(self, data):

        # Define message
        msg_yolo = yolomsg()

        # list initialization
        cls_list = []
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        score_list = []
        detected_obj = []
        detections = None

        if self.cnt % 5 == 0 and self.start_flag == 1: # Downsample

            prev_time = time.time()

            np_arr = np.fromstring(data.data, np.uint8)
            cv_image2_temp = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # YOLO
            cv_image2 = cv2.cvtColor(cv_image2_temp, cv2.COLOR_BGR2RGB)
            cv_image = Image.fromarray(cv_image2)

            input_transform = data_T(cv_image)

            input_transform,_ = pad_to_square(input_transform,0)

            input_transform = input_transform.unsqueeze(0).to(self.device)
            input_tensor = F.interpolate(input_transform,(self.opt.img_size,self.opt.img_size),mode='nearest')

            classes = load_classes(self.opt.class_path)  # Extracts class labels from file

            Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            # Perform inference
            with torch.no_grad():
                detections = self.model(input_tensor)
                detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)

            img = cv_image2

            cv_image2_t = cv2.resize(cv_image2_temp,(int(cv_image2_temp.shape[1]/4),int(cv_image2_temp.shape[0]/4)),cv2.INTER_NEAREST) # Image for visualization


            # Draw bounding boxes and labels of detections
            if detections[0] is not None:

                # Rescale boxes to original image
                detections = rescale_boxes(detections[0], self.opt.img_size,img.shape[:2])

                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)

                msg_yolo.num = len(detections)


                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print("\t+ Label: %s, Conf: %.3f" % (classes[int(cls_pred)], cls_conf.item()))

                    cv2.rectangle(cv_image2_t, (x1/4, y1/4), (x2/4, y2/4), (0,0,255), 2)
                    cv2.putText(cv_image2_t,"%s, %.3f" % (classes[int(cls_pred)], cls_conf.item()), (x1/4 + 20, y1/4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

                    cls_list.append(classes[int(cls_pred)])
                    x1_list.append(x1)
                    x2_list.append(x2)
                    y1_list.append(y1)
                    y2_list.append(y2)
                    score_list.append(cls_conf.item())

                print(cls_list)
            cv2.imshow('yolo_result', cv_image2_t)
            cv2.waitKey(1)

            # Publish message
            msg_yolo.className = cls_list
            msg_yolo.x1 = x1_list
            msg_yolo.x2 = x2_list
            msg_yolo.y1 = y1_list
            msg_yolo.y2 = y2_list
            msg_yolo.score = score_list

            self.pub.publish(msg_yolo)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Inference Time: %s" % (inference_time))

            print("\n")

        self.cnt += 1
        if self.cnt == 5:
            self.cnt = 0


def main(args):

    rospy.init_node('yolo_detection_node', anonymous=False)

    cnn = yolo_detection()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
