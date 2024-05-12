import os.path
import sys
import threading

import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QAbstractItemView, QTableWidgetItem, QFileDialog, QMessageBox

from ui.main_ui import Ui_MainWindow as mainWindow
import numpy as np
import time

from ui_controller import Detect_ROAD

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPalette, QBrush
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

class Main_Window(QtWidgets.QMainWindow, mainWindow):

    def __init__(self):
        super(Main_Window, self).__init__()
        self.setupUi(self)
        self.isStoped = False
        self.fps,self.w,self.h=0,0,0
        self.ispic = False

        self.initbtn()
        self.hasInit = False
        self.currentImg = None
        self.instance = Detect_ROAD()


    def initbtn(self):
        self.pushButton_pic.clicked.connect(self.pic_select)
        self.pushButton_video.clicked.connect(self.video_select)
        self.pushButton_camera.clicked.connect(self.camera_select)
        self.pushButton_stop.clicked.connect(self.stop_detect)



    def pic_select(self):
        #self.realdetect.hasStop = True
        tmp, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if tmp is None or tmp == "":
            return
        self.imgName = tmp
        self.ispic = True
        self.startDetect(self.instance.detect,True)


    def video_select(self):
        #self.realdetect.hasStop = True
        tmp, imgType = QFileDialog.getOpenFileName(self, "打开视频", "", " All Files(*)")
        if tmp is None or tmp == "":
            return
        self.imgName = tmp
        self.ispic = False
        self.startDetect(self.getvideo,False)

    def camera_select(self):
        #self.realdetect.hasStop = True
        self.imgName = 0
        self.ispic = False
        self.startDetect(self.getvideo,False)

    def startDetect(self,target,isPic):
        #self.realdetect.hasStop = False
        if isPic:
            self.detectThread = threading.Thread(target=target,args=(cv2.imread(self.imgName),self.showImage))
        else:
            self.detectThread = threading.Thread(target=target, args=(self.imgName,self.showImage))
        self.detectThread.start()

    def getvideo(self,img,callback):
        cap = cv2.VideoCapture(img )
        while not self.isStoped:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            self.fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
            self.w, self.h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
            self.instance.detect(frame,callback)


    def stop_detect(self):
        self.isStoped= True
        self.imgName = None

    def realShow(self,isInit):
        if self.currentImg is None:
            return
        if isInit:
            self.currentImg = cv2.resize(self.currentImg,(1200,800))
        frame = cv2.cvtColor(self.currentImg, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        if not isInit:
            width = self.label_2.width()
            height = self.label_2.height()
        else:
            width, height = 824, 568
        scale = max(width, height) / max(frame_w, frame_h)
        frame_h, frame_w = int(frame_h * scale), int(frame_w * scale)
        frame = cv2.cvtColor(self.currentImg, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(frame_w,frame_h))
        frame = cv2.copyMakeBorder(frame, int(np.maximum((height - frame_h) / 2, 0)),  # up
                                   int(np.maximum((height - frame_h) / 2, 0)),  # down
                                   int(np.maximum((width - frame_w) / 2, 0)),  # left
                                   int(np.maximum((width - frame_w) / 2, 0)),  # right
                                   cv2.BORDER_CONSTANT, value=[88, 155, 255])
        img = QImage(
            frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        del frame
        self.label_2.setPixmap(QPixmap.fromImage(img))


    def showImage(self,image,info,isInit):
        self.currentImg = image
        self.realShow(isInit)


    # 点击关闭按钮
    def closeEvent(self, e):
        self.isStoped= True
        self.imgName = None

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Ui_Main = Main_Window()
    Ui_Main.show()
    sys.exit(app.exec_())
