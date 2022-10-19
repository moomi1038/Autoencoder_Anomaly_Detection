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
import platform
from time import sleep, time

import module.record_module as record_module
import module.train_module as train_module
import module.test_module as test_module
import module.validation_module as validation_module
import module.serial_module as serial_module

with open("param.yaml") as f: 
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
    "RECORD_DATA" : None,
    "NOW" : QDate.currentDate(),
    "STYLE_NORMAL_WHITE" : "border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(255, 255, 255);",
    "STYLE_ABNORMAL_WHITE" : "border-width: 2px; border-radius: 10px; background-color: rgb(230, 5, 5); color: rgb(255, 255, 255);",
    "STYLE_ING_WHITE" : "border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(255, 255, 255);",
    "STYLE_NORMAL_BLACK" : "border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(0, 0, 0);",
    "STYLE_ABNORMAL_BLACK" : "border-width: 2px; border-radius: 10px; background-color: rgb(230, 5, 5); color: rgb(0, 0, 0);",
    "STYLE_ING_BLACK" : "border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(0, 0, 0);"
}

form_class = uic.loadUiType("main.ui")[0]

# m1 번방은 오류 없으면 항상 1 => 초록불
# m0 번방 1 => 빨간불
# 오류 발생시 => m3 번방에 1 => plc 초기화

class real_time_plc(QThread):
    send_data = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.check

    def run(self):
        while True:
            if (TOTAL_STATUS["PLC_STATUS"] == True) and (TOTAL_STATUS["NORMAL_STATUS"] == False):
                result = serial_module.port_error_message()
                if result:
                    print("Success")
                else:
                    print('Failed')

    def check(self):
        try:
            result, plf, port = serial_module.port_init()
            if result:
                TOTAL_DATA["PLF"] = plf
                TOTAL_DATA["PLC_PORT"] = port
            
        except:
            TOTAL_DATA["PLC_PORT"] = None
            TOTAL_STATUS["PLC_STATUS"] = False

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

                if param["GRAPH_TYPE"] == "Log Mel":
                    self.parent.setting_threshold_tot_label.setText(str(param["THRESHOLD_LOGMEL"]))

                if TOTAL_STATUS["NORMAL_STATUS"] == True:
                    self.parent.abnormal_check_label.setText("이상감지 정상")
                    self.parent.abnormal_check_label.setStyleSheet(TOTAL_DATA["STYLE_NORMAL_WHITE"])

                elif TOTAL_STATUS["NORMAL_STATUS"] == False:
                    self.parent.abnormal_check_label.setText("이상감지 비정상!")
                    self.parent.abnormal_check_label.setStyleSheet(TOTAL_DATA["STYLE_ABNORMAL_WHITE"])

                if TOTAL_STATUS["TRAIN_STATUS"] == False:
                    self.parent.deep_check_label.setText("딥러닝 정상")
                    self.parent.deep_check_label.setStyleSheet(TOTAL_DATA["STYLE_NORMAL_WHITE"])

                elif TOTAL_STATUS["TRAIN_STATUS"] == True:
                    self.parent.deep_check_label.setText("녹음중!")
                    self.parent.deep_check_label.setStyleSheet(TOTAL_DATA["STYLE_ING_WHITE"])

                if TOTAL_STATUS["TEST_STATUS"] == False and TOTAL_STATUS["TRAIN_STATUS"] == True:
                    self.parent.abnormal_check_label.setText("이상감지 시스템 \n 학습중!")
                    self.parent.abnormal_check_label.setStyleSheet(TOTAL_DATA["STYLE_ING_WHITE"])

                if TOTAL_STATUS["PLC_STATUS"] == True:
                    self.parent.module_check_label.setText("모듈연결 정상")
                    self.parent.module_check_label.setStyleSheet(TOTAL_DATA["STYLE_NORMAL_WHITE"])
                    
                elif TOTAL_STATUS["PLC_STATUS"] == False:
                    self.parent.module_check_label.setText("모듈연결 비정상!")
                    self.parent.module_check_label.setStyleSheet(TOTAL_DATA["STYLE_ABNORMAL_WHITE"])

                if (TOTAL_STATUS["TRAIN_STATUS"] == True) and (TOTAL_STATUS["REAL_TRAIN_STATUS"] == True):
                    self.parent.deep_check_label.setText("학습을 위해 \n 녹음 정지!")
                    self.parent.deep_check_label.setStyleSheet(TOTAL_DATA["STYLE_ABNORMAL_WHITE"])

                if (TOTAL_STATUS["REAL_TRAIN_STATUS"] == True) and (TOTAL_STATUS["VALIDATION_STATUS"] == True):
                    self.parent.abnormal_check_label.setText("이상감지 학습 \n 검증중!")
                    self.parent.abnormal_check_label.setStyleSheet(TOTAL_DATA["STYLE_ABNORMAL_WHITE"])
            except Exception as e:
                print("label e :", e)

class real_time_record(QThread):
    send_data = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()    

    def run(self):
        while True:
            if not TOTAL_STATUS["REAL_TRAIN_STATUS"]:
                if TOTAL_STATUS["RECORD_STATUS"]:
                    try:
                        TOTAL_DATA["RECORD_DATA"] = record_module.time_recording(TOTAL_STATUS["RECORD_STATUS"],param["GRAPH_TYPE"])

                        self.send_data.emit(TOTAL_DATA["RECORD_DATA"])

                    except Exception as e:
                        print("record e : ", e)

class real_time_test(QThread):
    send_data = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()    

    def run(self):
        count = 0
        while True:
            try:
                if (TOTAL_STATUS["TEST_STATUS"] == True) and (TOTAL_STATUS["TRAIN_STATUS"] == False):
                    if TOTAL_DATA["RECORD_DATA"] is not None:
                        start = time()

                        result = test_module.test_run(TOTAL_DATA["RECORD_DATA"],param["GRAPH_TYPE"],param["THRESHOLD_LOGMEL"])

                        TOTAL_DATA["RECORD_DATA"] = None

                        if not result:
                            count +=1
                            if count > param["PATIENCE"]:
                                TOTAL_STATUS["NORMAL_STATUS"] = False
                                TOTAL_STATUS["TEST_STATUS"] = False
                                count = 0
                                self.send_data.emit(TOTAL_STATUS["NORMAL_STATUS"])
                        else:
                            count = 0

                        end = time()
                        print("PATIENCE COUNT : ", count)

                if TOTAL_STATUS["TEST_STATUS"] == False:
                    count = 0

            except Exception as e:
                print("test e ",e)

class real_time_train(QThread): 
    send_data = pyqtSignal(object)
    
    def __init__(self,parent):
        super().__init__(parent)    
        self.parent = parent

    def run(self):
        csv_len = (param["TRAIN_TOTAL"] / param["PYAUDIO_SECONDS"])
        count = 0
        while True:
            if TOTAL_STATUS["TRAIN_STATUS"]:
                try:
                    if TOTAL_DATA["RECORD_DATA"] is not None:
                        print(count)
                        if count < csv_len:
                            record_module.save_csv(TOTAL_DATA["RECORD_DATA"], count, param["GRAPH_TYPE"])
                            count += 1
                            progress_value = int(count / csv_len * 100)
                            self.parent.progressBar.setValue(progress_value)
                            TOTAL_DATA["RECORD_DATA"] = None

                        else:
                            TOTAL_STATUS["REAL_TRAIN_STATUS"] = True
                            train_module.train_run(param["GRAPH_TYPE"])
                            TOTAL_STATUS["VALIDATION_STATUS"] = True
                            result = validation_module.validation_run(param["GRAPH_TYPE"])
                            self.send_data.emit(result)
                            TOTAL_STATUS["REAL_TRAIN_STATUS"] = False
                            TOTAL_STATUS["VALIDATION_STATUS"] = False

                except Exception as e:
                    print("train e : ", e)

class gui(QMainWindow, form_class):
    def __init__(self):
        super().__init__() # __init__ is parent class

        self.setupUi(self)
        self.UIStyle()
        self.UIinit()

        self.real_record = real_time_record()
        self.real_record.send_data.connect(self.graph)
        self.real_record.start()

        self.real_test = real_time_test()
        self.real_test.send_data.connect(self.error_check)
        self.real_test.start()

        self.real_train = real_time_train(self)
        self.real_train.send_data.connect(self.train_check)

        self.real_label = real_time_label(self)
        self.real_label.start()

        self.real_plc = real_time_plc()
        self.real_plc.start()

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.mat.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time frame")
        self.ax.set_ylabel("Frequency")

    def graph(self, data):
        try:
            s = time()

            self.ax.cla()
            self.ax.set_xlabel("Time frame")
            self.ax.set_ylabel("Frequency")
            if data is not None:
                if param["GRAPH_TYPE"] == 'Log Mel':
                    graph = librosa.display.specshow(data, sr=param["AUDIO_SAMPLERATE"],x_axis='time',y_axis = 'mel', ax = self.ax, cmap="coolwarm",clim=(0,100))
                    self.canvas.draw_idle()
                    self.canvas.flush_events()

                    del graph
                

            e = time()
            print("graph_time  : ",e-s)

        except Exception as e:
            print("graph e : ", e)


#### BUTTON ####
    def close(self):
        QApplication.instance().quit()

    def restart(self):
        os.execl(sys.executable, sys.executable, *sys.argv)


#### UINIT ####
    def UIinit(self):
        #### CLICKED ####
        self.close_button.clicked.connect(self.close)
        self.check_restart_button.clicked.connect(self.restart)
        self.check_deep_button.clicked.connect(self.deep_check)
        self.check_module_button.clicked.connect(self.module_check)
        self.check_train_button.clicked.connect(lambda:self.train_check(True))

        #### SLIDER ####
        self.setting_threshold_slider.valueChanged[int].connect(self.threshold_setting)
        # self.setting_train_slider.valueChanged[int].connect(lambda : self.train_total('label', value))
        self.setting_train_slider.sliderReleased.connect(lambda: self.train_total())
        self.setting_patience_slider.valueChanged[int].connect(self.patience_setting)

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
