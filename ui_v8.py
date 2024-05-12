import argparse
import random
import time
import numpy as np

from v8.ultralytics.nn.autobackend import AutoBackend
from v8.ultralytics.yolo.data.augment import LetterBox

import torch

from v8.ultralytics.yolo.utils import  ROOT, ops
from v8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from v8.ultralytics.yolo.utils.torch_utils import select_device


class Detect():
    def __init__(self):
        self.weights = './v8/ultralytics/yolo/v8/detect/runs/detect/train/weights/best.pt'
        self.img_size = 640
        self.conf_thres =0.1
        self.iou_thres =0.4

        self.device = select_device("0")
        self.half = self.device.type != 'cpu'

        self.model = AutoBackend(self.weights, device=self.device, dnn=False, fp16=self.half)
        self.model.to(self.device).eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # Run inference
        img_init = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img_init.half() if self.half else img_init) if self.device.type != 'cpu' else None  # run once
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=False,
                                        max_det=1000)

        for i, pred in enumerate(preds):
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds


    def detect(self,img0,callback= None):
        img0 = img0.copy()
        self.annotator = Annotator(img0, line_width=2 )

        im = LetterBox(self.img_size, self.model.pt, stride=self.model.stride)(image=img0)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        img = torch.from_numpy(im).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        timestart = time.time()
        preds = self.model(img, augment=False, visualize=False)
        preds = self.postprocess(preds, img, img0) # nms
        counter = 0
        cordinfo = {}
        for *xyxy, conf, cls in reversed(preds[0]):
            label = f'{self.names[int(cls)]} {conf:.2f}'
            counter+=1
            cordinfo[str(counter)] = [str(int(xyxy[0])),str(int(xyxy[1])),str(int(xyxy[2])),str(int(xyxy[3]))]

            self.annotator.box_label(xyxy, label, color=colors(int(cls) , True))
        timeend = time.time()

        if callback is not None:
            callback(self.annotator.im,"",False)
        return self.annotator.im#,infos

import cv2
if __name__ == "__main__":
    dd = Detect()
    xxx = dd.detect(cv2.imread("./demo/test.jpg"))
    cv2.imshow("xx",xxx)
    cv2.waitKey(0)