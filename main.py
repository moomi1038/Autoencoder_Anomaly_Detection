import os
from PyQt6 import uic
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import QDate, QThread, pyqtSignal,Qt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import librosa
import librosa.display
import sys
import yaml
import psutil
import tensorflow as tf
from time import sleep, time

import module.record_module as record_module
import module.train_module as train_module
import module.test_module as test_module
import module.validation_module as validation_module
import module.serial_module as serial_module
import module.keras_model as keras_module

path = os.getcwd()
param_path = os.path.join(path,"param.yaml")

with open(param_path) as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

TOTAL_STATUS = {
    "TRAIN_STATUS" : False, # True : training , False : Trained(default)
    "REAL_TRAIN_STATUS" : False,
    "VALIDATION_STATUS" : False,
    "TEST_STATUS" : True, # True : testing, False : Not Testing(default)
    "RECORD_STATUS" : True, # True : recording, False : not recording
    "NORMAL_STATUS" : True, # True : normal, False : abnormal
    "PLC_STATUS" : False # True : plc connected, False : not connected
}

TOTAL_DATA = {
    "PLC_PORT" : None, # PLC PORT
    "PLF" : None,
    "NOW" : QDate.currentDate(),
    "PATIENCE" : 0,
    "TRAIN_TOTAL" : 0
}
TOTAL_STYLE = {
    "STYLE_NORMAL_WHITE" : "border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(255, 255, 255);",
    "STYLE_ABNORMAL_WHITE" : "border-width: 2px; border-radius: 10px; background-color: rgb(230, 5, 5); color: rgb(255, 255, 255);",
    "STYLE_ING_WHITE" : "border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(255, 255, 255);",
    "STYLE_NORMAL_BLACK" : "border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(0, 0, 0);",
    "STYLE_ABNORMAL_BLACK" : "border-width: 2px; border-radius: 10px; background-color: rgb(230, 5, 5); color: rgb(0, 0, 0);",
    "STYLE_ING_BLACK" : "border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(0, 0, 0);"
}

form_class = uic.loadUiType("main.ui")[0]

class real_time_label(QThread):
    def __init__(self,parent):
        super().__init__(parent)    
        self.parent = parent

    def run(self):
        while True:
            try:
                self.parent.statusBar().showMessage(f" SERIAL : {TOTAL_DATA['PLC_PORT']} , RECORD STATUS : {TOTAL_STATUS['RECORD_STATUS']} , TRAIN STATUS : {TOTAL_STATUS['TRAIN_STATUS']} , Validation STATUS : {TOTAL_STATUS['VALIDATION_STATUS']} ,REAL TRAIN STATUS : {TOTAL_STATUS['REAL_TRAIN_STATUS']} ,TEST STATUS : {TOTAL_STATUS['TEST_STATUS']}")

                self.parent.cpu_usage.setText(str(psutil.cpu_percent(interval = 1)) + " %")
                self.parent.ram_usage.setText(f'{str(psutil.virtual_memory().used / 1024**3):.4}' + " GB")
                self.parent.time_label.setText(QDate.currentDate().toString(Qt.DateFormat.ISODate))
                self.parent.setting_threshold_tot_label.setText(str(param["THRESHOLD_STFT"]))
                self.parent.setting_patience_label.setText(str(param["PATIENCE"]))
                self.parent.patience_slider.setMaximum(param["PATIENCE"])
                self.parent.patience_slider.setValue(TOTAL_DATA["PATIENCE"])

                if TOTAL_STATUS["NORMAL_STATUS"] == True:
                    self.parent.abnormal_check_label.setText("이상감지 정상")
                    self.parent.abnormal_check_label.setStyleSheet(TOTAL_STYLE["STYLE_NORMAL_WHITE"])

                elif TOTAL_STATUS["NORMAL_STATUS"] == False:
                    self.parent.abnormal_check_label.setText("이상감지 비정상!")
                    self.parent.abnormal_check_label.setStyleSheet(TOTAL_STYLE["STYLE_ABNORMAL_WHITE"])

                if TOTAL_STATUS["TRAIN_STATUS"] == False:
                    self.parent.deep_check_label.setText("딥러닝 정상")
                    self.parent.deep_check_label.setStyleSheet(TOTAL_STYLE["STYLE_NORMAL_WHITE"])

                elif TOTAL_STATUS["TRAIN_STATUS"] == True:
                    self.parent.deep_check_label.setText("녹음중!")
                    self.parent.deep_check_label.setStyleSheet(TOTAL_STYLE["STYLE_ING_WHITE"])

                if TOTAL_STATUS["TEST_STATUS"] == False and TOTAL_STATUS["TRAIN_STATUS"] == True:
                    self.parent.abnormal_check_label.setText("이상감지 시스템 \n 학습중!")
                    self.parent.abnormal_check_label.setStyleSheet(TOTAL_STYLE["STYLE_ING_WHITE"])

                if TOTAL_STATUS["PLC_STATUS"] == True:
                    self.parent.module_check_label.setText("모듈연결 정상")
                    self.parent.module_check_label.setStyleSheet(TOTAL_STYLE["STYLE_NORMAL_WHITE"])
                    
                elif TOTAL_STATUS["PLC_STATUS"] == False:
                    self.parent.module_check_label.setText("모듈연결 비정상!")
                    self.parent.module_check_label.setStyleSheet(TOTAL_STYLE["STYLE_ABNORMAL_WHITE"])

                if (TOTAL_STATUS["TRAIN_STATUS"] == True) and (TOTAL_STATUS["REAL_TRAIN_STATUS"] == True):
                    self.parent.deep_check_label.setText("학습을 위해 \n 녹음 정지!")
                    self.parent.deep_check_label.setStyleSheet(TOTAL_STYLE["STYLE_ABNORMAL_WHITE"])

                if (TOTAL_STATUS["REAL_TRAIN_STATUS"] == True) and (TOTAL_STATUS["VALIDATION_STATUS"] == True):
                    self.parent.abnormal_check_label.setText("이상감지 학습 \n 검증중!")
                    self.parent.abnormal_check_label.setStyleSheet(TOTAL_STYLE["STYLE_ABNORMAL_WHITE"])
            except Exception as e:
                print("label e :", e)

class real_time_record(QThread):
    send_data = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()    
        try: 
            temp = os.path.join(path, param["DIR_NAME_MODEL"])
            self.model = keras_module.load_model(temp)

            temp = os.path.join(path, param["DENOISE_MODEL"])
            denoise_model = tf.saved_model.load(temp)
            self.infer = denoise_model.signatures["serving_default"]
            
        except Exception as e:
            print("Model load e : ", e)

    def run(self):
        while True:
            if TOTAL_STATUS["RECORD_STATUS"]:
                try:
                    data = record_module.time_recording(param["AUDIO_SAMPLERATE"],param["PYAUDIO_CHUNK"],param["LIBROSA_N_FFT"], self.infer)
                    
                    self.send_data.emit(data)

                    if TOTAL_STATUS["TEST_STATUS"]:
                        result = test_module.test_run(data, self.model,param["THRESHOLD_STFT"])
                        if not result:
                            TOTAL_DATA["PATIENCE"] +=1
                            if TOTAL_DATA["PATIENCE"] >= param["PATIENCE"]:
                                print("ERROR OCCUR")
                        else:
                            TOTAL_DATA["PATIENCE"] = 0

                except Exception as e:
                    print("record e : ", e)        

class gui(QMainWindow, form_class):
    def __init__(self):
        super().__init__() # __init__ is parent class

        self.setupUi(self)
        self.UIStyle()
        self.UIinit()

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.mat.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time frame")
        self.ax.set_ylabel("Frequency")

        self.real_label = real_time_label(self)
        self.real_label.start()

        self.real_record = real_time_record()
        self.real_record.send_data.connect(self.graph)
        self.real_record.start()

    def graph(self, data):
        try:
            s = time()
            self.ax.cla()
            self.ax.set_xlabel("Time frame")
            self.ax.set_ylabel("Frequency")
            if data is not None:
                librosa.display.specshow(data, hop_length= param["LIBROSA_N_FFT"] // 4 ,sr=param["AUDIO_SAMPLERATE"],x_axis='time',y_axis = 'linear', ax = self.ax, cmap="coolwarm")
                self.canvas.draw_idle()
                self.canvas.flush_events()
                
            e = time()
            print("graph_time  : ",e-s)

        except Exception as e:
            print("graph e : ", e)

    def error_check(self, boolean):
        try:
            if boolean:
                TOTAL_STATUS["NORMAL_STATUS"] = True
            else:
                TOTAL_STATUS["NORMAL_STATUS"] = False
                TOTAL_STATUS["TEST_STATUS"] = False
        except Exception as e:
            print("error_check e : ", e)

#### VALUE ####
    # def train_setting(self,value):
    #     self.setting_train_label.setText(str(value))

    # def threshold_setting(self,value):
    #     print(value)
    #     self.setting_threshold_tot_label.setText(str(value))

    # def patience_setting(self,value):
    #     self.setting_patience_label.setText(str(value))

#### BUTTON ####
    def close(self):
        QApplication.instance().quit()

    def restart(self):
        os.execl(sys.executable, sys.executable, *sys.argv)

#### SLIDER ####
    def train_total(self):
        try:
            TOTAL_STATUS["TRAIN_STATUS"] = False
            value = self.setting_train_slider.value()
            self.setting_train_label.setText(str(value))
            param["TRAIN_TOTAL"] = value * 60
            TOTAL_DATA["TRAIN_TOTAL"] = value * 60 // 3
            with open(param_path, 'w') as file:
                yaml.dump(param, file, default_flow_style=False)

        except Exception as e:
            print("train_total e : ",e )
            
    def threshold_total(self,value):
        try:
            TOTAL_STATUS["TEST_STATUS"] = False
            THRESHOLD = round(float(param["THRESHOLD_LOGMEL"] + value))
            self.setting_threshold_tot_label.setText(str(THRESHOLD))
            param["THRESHOLD_LOGMEL"] = THRESHOLD

            with open(param_path, 'w') as file:
                yaml.dump(param, file, default_flow_style=False)

            self.setting_threshold_slider.setValue(0)
            print(param["THRESHOLD_LOGMEL"])
            TOTAL_STATUS["TEST_STATUS"] = True

        except Exception as e:
            print("threshold_setting e :", e)

    def patience_total(self):
        try:
            TOTAL_STATUS["TEST_STATUS"] = False
            value = self.setting_train_slider.value()
            param["PATIENCE"] = value

            with open(param_path, 'w') as file:
                yaml.dump(param, file, default_flow_style=False)

            self.setting_patience_label.setText(str(value))

            if not TOTAL_STATUS["TRAIN_STATUS"]:
                TOTAL_STATUS["TEST_STATUS"] = True

        except Exception as e:
            print("patience_setting e : ", e)

    def deep_check(self):
        print("DFDF")
        self.tabWidget.setCurrentIndex(0)

    def module_check(self):
        print("DFDF")
        self.tabWidget.setCurrentIndex(0)

    def train_check(self, boolean):
        self.tabWidget.setCurrentIndex(0)
        TOTAL_STATUS["TRAIN_STATUS"] = boolean
        if TOTAL_STATUS["TRAIN_STATUS"]:
            TOTAL_STATUS["TEST_STATUS"] = False
            self.real_train.start()

        else:
            TOTAL_STATUS["TRAIN_STATUS"] = False
            TOTAL_STATUS["TEST_STATUS"] = True
            for file in os.scandir(param["DIR_NAME_TRAIN_LOGMEL"]):
                os.remove(file.path)

#### UINIT ####

    def UIinit(self):
        #### CLICKED ####
        self.close_button.clicked.connect(self.close)
        self.check_restart_button.clicked.connect(self.restart)
        self.check_deep_button.clicked.connect(self.deep_check)
        self.check_module_button.clicked.connect(self.module_check)
        self.check_train_button.clicked.connect(lambda:self.train_check(True))

        #### SLIDER ####
        self.setting_train_slider.sliderReleased.connect(lambda: self.train_total)
        self.setting_threshold_slider.sliderReleased.connect(lambda: self.threshold_total)
        self.setting_patience_slider.sliderReleased.connect(lambda: self.patience_total)

        # self.setting_threshold_slider.valueChanged[int].connect(self.threshold_setting)
        # self.setting_train_slider.valueChanged[int].connect(self.train_setting)
        # self.setting_patience_slider.valueChanged[int].connect(self.patience_setting)

    def UIStyle(self):
        #### ICON #####
        self.tabWidget.setTabIcon(0,QIcon("./resource/pic_dashboard.png"))
        self.tabWidget.setTabIcon(1,QIcon("./resource/pic_check.png"))
        self.tabWidget.setTabIcon(2,QIcon("./resource/pic_setting.png"))
        self.close_button.setIcon(QIcon("./resource/pic_close.png"))

        #### LABEL ####
        self.time_label.setText(TOTAL_DATA["NOW"].toString(Qt.DateFormat.ISODate))
        self.statusbar.setStyleSheet("QStatusBar{padding-left:8px;color:white}")
        self.setting_patience_label.setText(str(param["PATIENCE"]))
        self.setting_train_label.setText(str(int(param["TRAIN_TOTAL"]/60)))

        #### SLIDER ####
        self.setting_patience_slider.setValue(int(param["PATIENCE"]))
        self.setting_train_slider.setValue(int(param["TRAIN_TOTAL"]/60))

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = gui()
    myWindow.show()
    app.exec()