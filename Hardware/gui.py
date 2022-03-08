import sys
from tkinter.tix import Tree
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

import cv2
import numpy as np
import time

import bpm
from imutil import WebCamVideoStream

styling = open('stylesheet.css').read()

class MainProgram(QtWidgets.QWidget):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.layout = QtWidgets.QGridLayout()

        self.feed_text_default = '<b>Harap tekan tombol Start untuk memulai perhitungan detak jantung.</b>'
        self.feed_label = QtWidgets.QLabel(text = self.feed_text_default, objectName = 'feed_label')
        self.feed_label.setWordWrap(True)
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setStyleSheet(styling)
        self.layout.addWidget(self.feed_label, 1, 1, 4, 1)

        info_text = '<b>Pengukuran Detak Jantung Nirkontak</b><br>Oleh:Tim Contactless HRM ITB'
        self.info_label = QtWidgets.QLabel(text= info_text, objectName = 'info_label')
        self.info_label.setStyleSheet(styling)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.info_label, 1, 2, 1, 1)

        self.result_label = QtWidgets.QLabel(objectName = 'result_label')
        self.result_label.setStyleSheet(styling)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label, 2, 2, 1, 1)

        self.start_btn = QtWidgets.QPushButton(text = 'Start', objectName = 'start_btn')
        self.start_btn.setStyleSheet(styling)
        self.start_btn.clicked.connect(self.start)
        self.layout.addWidget(self.start_btn, 3, 2, 1, 1)

        self.stop_btn = QtWidgets.QPushButton(text = 'Stop' , objectName = 'stop_btn')
        self.stop_btn.setStyleSheet(styling)
        self.stop_btn.clicked.connect(self.stop)
        self.layout.addWidget(self.stop_btn, 4, 2, 1, 1)

        self.setLayout(self.layout)
    
    def start(self):
        # Executing Slot
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.bpm_signal.connect(self.update_bpm)
        self.thread.start()

    def stop(self):
        self.thread.change_pixmap_signal.disconnect()
        self.thread.bpm_signal.disconnect()
        self.thread._run_flag = False
        self.thread.stream.stop()
        self.feed_label.clear()
        self.feed_label.setText(self.feed_text_default)
        print('Calculation is stopped')

    QtCore.pyqtSlot(float)
    def update_bpm(self, bpm):
        if bpm < 0:
            self.result_label.setText('<b>Wait...</b>')    
        else:
            self.result_label.setText('Nilai detak jantung anda:<br>' + '<b>{}</b>'.format(str(bpm)))

    QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.feed_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)
    bpm_signal = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.time = time.time()
    
    def run(self):
        r = []
        g = []
        b = []
        arr_time = []
        list_bpm = []
        window_size = 100
        number_frame = 0
        count = 0
        found_face = False
        tracker_init = False
        face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

        self.stream = WebCamVideoStream(src=0).start()
        while self._run_flag:
            img = self.stream.read()
            grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if found_face and tracker_init:
                found_face, face_box = tracker.update(img)
            
            if found_face==False:
                tracker_init = False
                faces = face_cascade.detectMultiScale(
                        grayscale_img, 
                        scaleFactor     = 1.2,
                        minNeighbors    = 5,
                        minSize         = (30,30)
                    )
                found_face = len(faces) > 0
            
            if found_face and not tracker_init:
                face_box = faces[0]
                tracker = cv2.TrackerMOSSE_create()
                tracker.init(img, tuple(face_box))
                tracker_init = True
            
            if found_face:
                try:
                    #ROI dahi
                    x1 = int(face_box[0]+0.2*face_box[2])
                    y1 = int(face_box[1] + 0*face_box[3])
                    x2 = int(face_box[0] + 0.8*face_box[2])
                    y2 = int(face_box[1] + 1*face_box[3])

                    p1 = (x1,y1)
                    p2 = (x2,y2)
                    cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
                    clipped_roi =img[y1:y2, x1:x2]

                    color_channel = np.mean(clipped_roi, axis=(0,1))
                    R = np.nan_to_num(color_channel[2])
                    G = np.nan_to_num(color_channel[1])
                    B = np.nan_to_num(color_channel[0])

                    r.append(R)
                    g.append(G)
                    b.append(B)
                    number_frame += 1
                    image = cv2.resize(img, None, None, fx=2, fy=2)
                    arr_time.append(time.time())
                    # Emit signal pyqt slot for video
                    self.change_pixmap_signal.emit(image)
                    if (number_frame>window_size):
                        fs = int(window_size/(arr_time[count+window_size]-arr_time[count]))
                        r_process = bpm.cut_window(r,count,count+window_size)
                        g_process = bpm.cut_window(g,count,count+window_size)
                        b_process = bpm.cut_window(b,count,count+window_size)
                        bpm_res = bpm.process_HR(r_process, g_process, b_process, fs)
                        list_bpm.append(bpm_res)
                        self.bpm_signal.emit(int(np.mean(list_bpm)))
                        count +=1
                    else:
                        # Emit signal pyqt slot for BPM
                        self.bpm_signal.emit(-1)
                except:
                    pass

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName('Contactless Heart Rate Measurement')
    program = MainProgram()
    program.setFixedSize(1440, 900)
    program.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
