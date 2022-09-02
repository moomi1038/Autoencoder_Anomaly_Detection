# from ast import Param
import os
from PyQt6 import uic
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import serial
import serial.tools.list_ports as sp
import librosa
import librosa.display
import sys
import yaml
import psutil
import platform
from time import sleep, time

import record
import train
import test
import validation

with open("param.yaml") as f: 
    param = yaml.load(f, Loader=yaml.FullLoader)

TRAIN_STATUS = False # True : training , False : Trained(default)
TEST_STATUS = True # True : testing, False : Not Testing(default)
RECORD_STATUS = True # True : recording, False : not recording
STATUS = True # True : normal, False : abnormal
MODULE_STATUS = True # True : plc connect False : disconnect
RECORD_DATA = None
TRAIN_TOTAL = 1200 # sec
GRAPH = "Log Mel"
NOW = QDate.currentDate()
PORT = str()
PLATFORM = str()
PLC_STATUS = False
PLC_BOOLEAN = True
TEST_BOOLEAN = True
PATIENCE = param["PATIENCE"]
THRESHOLD_LOGMEL = param["THRESHOLD_LOGMEL"]
form_class = uic.loadUiType("main.ui")[0]

# m1 번방은 오류 없으면 항상 1 => 초록불
# m0 번방 1 => 빨간불
# 오류 발생시 => m3 번방에 1 => plc 초기화

class real_time_plc(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        global PLATFORM
        global PORT
        global PLC_STATUS # plc 연결 성공 / 실패
        global PLC_BOOLEAN
        global ser
        self.parent = parent
        list = sp.comports()
        plf = platform.system()
        try:
            for i in list:
                tmp = str(i)
                tmp = tmp.split(" ")
                if plf == "Darwin":
                    tmp = tmp[0]
                
                    if "usbserial" in tmp:
                        ser = serial.Serial(port = tmp, baudrate = 9600, timeout = 1)
                        temp = '\x0501RSS0106%MW' + '003' +'\x04'
                        temp = temp.encode()
                        ser.write(temp)
                        temp_result = ser.readline().decode('ascii')

                        if temp_result[0] == '\x06':
                            PLC_STATUS = True
                            break

                        else:
                            PLC_STATUS = False

                elif plf == "Windows":
                    tmp = tmp[0]
                    if "COM" in tmp:
                        ser = serial.Serial(port = tmp, baudrate = 9600, timeout = 1)
                        temp = '\x0501RSS0106%MW' + '003' +'\x04'
                        temp = temp.encode()
                        ser.write(temp)
                        temp_result = ser.readline().decode('ascii')
                    
                        if temp_result[0] == '\x06':
                            PLC_STATUS = True
                            break
                        
                        else:
                            PLC_STATUS = False

            PORT = tmp
            PLATFORM = plf

        except:
            PLC_STATUS = False

    def run(self):
        global PORT
        global PLC_STATUS
        global STATUS
        global PLATFORM
        global ser
        global PLC_BOOLEAN
        while True:
            try:
                ser = serial.Serial(port = PORT, baudrate = 9600, timeout = 1)

                temp = '\x0501RSS0106%MW' + '003' +'\x04'
                temp = temp.encode()
                ser.write(temp)
                temp_result = ser.readline().decode('ascii')
                
                # sleep(0.1)
                if temp_result[-2] == "0":
                    PLC_STATUS = True
            except:
                PLC_STATUS = False

            if (STATUS == False) and (PLC_STATUS == True):
                if temp_result[-2] == "0":
                    # sleep(0.5)
                    ser.write(b'\x0501WSS0106%MW0030001\x04')
                    sleep(0.5)


class real_time_label(QThread):
    def __init__(self,parent):
        super().__init__(parent)    
        self.parent = parent

    def run(self):
        global STATUS
        global TRAIN_STATUS
        global PLC_STATUS
        while True:
            try:
                cpu = str(psutil.cpu_percent(interval = 1)) 
                ram = str(psutil.virtual_memory().used / 1024**3)
                self.parent.cpu_usage.setText(cpu + " %")
                self.parent.ram_usage.setText(f'{ram:.4}' + " GB")
                if GRAPH == "Log Mel":
                    self.parent.setting_threshold_tot_label.setText(str(param["THRESHOLD_LOGMEL"]))
                elif GRAPH == "STFT":
                    self.parent.setting_threshold_tot_label.setText(str(param["THRESHOLD_STFT"]))
                if STATUS == True:
                    self.parent.abnormal_check_label.setText("이상감지 정상")
                    self.parent.abnormal_check_label.setStyleSheet("border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(255, 255, 255);")
                elif STATUS == False:
                    self.parent.abnormal_check_label.setText("이상감지 비정상!")
                    self.parent.abnormal_check_label.setStyleSheet("border-width: 2px; border-radius: 10px; background-color: rgb(230, 5, 5); color: rgb(255, 255, 255);")

                if TRAIN_STATUS == False:
                    self.parent.deep_check_label.setText("딥러닝 정상")
                    self.parent.deep_check_label.setStyleSheet("border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(255, 255, 255);")

                elif TRAIN_STATUS == True:
                    self.parent.deep_check_label.setText("녹음중!")
                    self.parent.deep_check_label.setStyleSheet("border-width: 2px; border-radius: 10px; background-color: rgb(230, 5, 5); color: rgb(255, 255, 255);")

                if TEST_STATUS == False and TRAIN_STATUS == True:
                    self.parent.abnormal_check_label.setText("이상감지 학습중!")
                    self.parent.abnormal_check_label.setStyleSheet("border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(255, 255, 255);")

                if PLC_STATUS == True:
                    self.parent.module_check_label.setText("모듈연결 정상")
                    self.parent.module_check_label.setStyleSheet("border-width: 2px; border-radius: 10px; background-color: rgb(5, 172, 230); color: rgb(255, 255, 255);")
                
                elif PLC_STATUS == False:
                    self.parent.module_check_label.setText("모듈연결 비정상!")
                    self.parent.module_check_label.setStyleSheet("border-width: 2px; border-radius: 10px; background-color: rgb(230, 5, 5); color: rgb(255, 255, 255);")
            
            except Exception as e:
                print(e)

class real_time_record(QThread):
    send_data = pyqtSignal(object)
    
    def __init__(self,parent):
        super().__init__(parent)    
        self.parent = parent

    def run(self):
        global RECORD_STATUS
        global TEST_STATUS
        global TRAIN_STATUS
        global GRAPH
        global RECORD_DATA
        global PORT
        while RECORD_STATUS:
            try:
                self.parent.statusBar().showMessage(f"SERIAL : {PORT} , RECORD STATUS : {RECORD_STATUS} , TRAIN STATUS : {TRAIN_STATUS} , TEST STATUS : {TEST_STATUS}")
                RECORD_DATA = record.time_recording(RECORD_STATUS,GRAPH)
                self.send_data.emit(RECORD_DATA)
            except Exception as e:
                print(e)

    def stop(self):
        self.exit()
        self.wait(300)

class real_time_test(QThread):
    send_data = pyqtSignal(object)
    
    def __init__(self,parent):
        super().__init__(parent)    
        self.parent = parent

    def run(self):
        global STATUS
        global TEST_STATUS
        global TEST_BOOLEAN
        global RECORD_DATA
        global TRAIN_STATUS
        global THRESHOLD_LOGMEL
        global PATIENCE
        global count 
        count = 0
        while TEST_BOOLEAN:
            try:
                if TEST_STATUS:
                    if not TRAIN_STATUS:
                        if RECORD_DATA is not None:
                            start = time()
                            result = test.test_run(RECORD_DATA,GRAPH,THRESHOLD_LOGMEL)

                            RECORD_DATA = None

                            if not result:
                                count +=1
                                if count == PATIENCE:
                                    STATUS = False
                                    TEST_STATUS = False
                            else:
                                count = 0
                            end = time()
                            print("ELAPSED TIME : ", end - start)
                            print("PATIENCE COUNT : ", count)
                            self.send_data.emit(STATUS)
            except Exception as e:
                print(e)

    def stop(self):
        self.exit()
        self.wait(300)

class real_time_train(QThread): # 해야함
    send_data = pyqtSignal(object)
    
    def __init__(self,parent):
        super().__init__(parent)    
        self.parent = parent

    def run(self):
        global RECORD_STATUS
        global GRAPH
        global TEST_STATUS
        global TRAIN_STATUS
        global RECORD_DATA
        csv_len = (param["TRAIN_TOTAL"] / param["PYAUDIO_SECONDS"])
        count = 0
        while TRAIN_STATUS:
            try:
                TEST_STATUS = False
                if RECORD_DATA is not None:
                    if count != csv_len:
                        train.save_csv(RECORD_DATA, count, GRAPH)
                        count += 1
                        progress_value = int(count / csv_len * 100)
                        self.parent.progressBar.setValue(progress_value)

                        RECORD_DATA = None
                        
                    else:    
                        train.train_run(GRAPH)
                        result = validation.validation_run(GRAPH)
                        TRAIN_STATUS = False
                        self.send_data.emit(result)
                        # TEST_STATUS = True
            except Exception as e:
                print(e)
                
    def stop(self):
        self.exit()
        self.wait(200)

class gui(QMainWindow, form_class):
    def __init__(self):
        super().__init__() # __init__ is parent class

        self.setupUi(self)
        self.UIStyle()
        self.UIinit()

        self.real_record = real_time_record(self)
        self.real_record.send_data.connect(self.graph)
        self.real_record.start()

        self.real_test = real_time_test(self)
        self.real_test.send_data.connect(self.error_check)
        self.real_test.start()

        self.real_train = real_time_train(self)
        self.real_train.send_data.connect(self.train_check)

        self.real_label = real_time_label(self)
        self.real_label.start()

        self.real_plc = real_time_plc(self)
        self.real_plc.start()
        
    def graph(self, data):
        global GRAPH

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        if self.mat.itemAt(0) is not None:
            self.mat.itemAt(0).widget().setParent(None)

        self.mat.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)

        if GRAPH == 'Log Mel':
            graph = librosa.display.specshow(data, sr=param["AUDIO_SAMPLERATE"],x_axis='time',y_axis = 'mel', ax = self.ax, cmap="coolwarm",clim=(0,100))
            self.ax.set_title("Log Mel Spectrogram (dB)")

        elif GRAPH == 'STFT':
            data = abs(data)
            graph = librosa.display.specshow(data, sr=param["AUDIO_SAMPLERATE"],x_axis='time',y_axis = 'linear', ax = self.ax, cmap="coolwarm",clim=(0,100))
            self.ax.set_title("STFT Spectrogram (dB)")

        self.ax.set_xlabel("Time frame")
        self.ax.set_ylabel("Frequency")
        self.fig.colorbar(graph, format="%+2.0f dB")

    def sec_setting(self):
        global GRAPH
        global TEST_STATUS
        global RECORD_DATA
        global param
        TEST_STATUS = False
        self.real_record.stop()
        time_sec = self.setting_sec_combobox.currentText()

        param["PYAUDIO_SECONDS"] = int(time_sec)
        with open('param.yaml', 'w') as file:
            yaml.dump(param, file, default_flow_style=False)
        
        RECORD_DATA = None
        self.real_record.start()
        TEST_STATUS = True

    def error_check(self, boolean):
        global STATUS
        if boolean:
            STATUS = True
        else:
            STATUS = False
            self.real_test.stop()
            

#### VALUE ####
    def train_total(self,value):
        global TRAIN_TOTAL
        TRAIN_TOTAL = int(value) * 60
        self.setting_train_label.setText(str(value))
        param["TRAIN_TOTAL"] = TRAIN_TOTAL
        with open('param.yaml', 'w') as file:
            yaml.dump(param, file, default_flow_style=False)

#### BUTTON ####
    def close(self):
        QApplication.instance().quit()

    def restart(self):
        os.execl(sys.executable, sys.executable, *sys.argv)

    def threshold_setting(self,value):
        global GRAPH
        global param
        global count
        global THRESHOLD_LOGMEL
        global TEST_STATUS
        TEST_STATUS = False
        count = 0
        if GRAPH == "Log Mel":
            param["THRESHOLD_LOGMEL"] = round(float(param["THRESHOLD_LOGMEL"] + value),1)
            self.setting_threshold_tot_label.setText(str(param["THRESHOLD_LOGMEL"]))

        elif GRAPH == "STFT":
            param["STFT"] = round(float(param["STFT"] + value),1)
            self.setting_threshold_tot_label.setText(str(param["THRESHOLD_STFT"]))

        with open('param.yaml', 'w') as file:
            yaml.dump(param, file, default_flow_style=False)

        THRESHOLD_LOGMEL = round(float(param["THRESHOLD_LOGMEL"]),1)
        self.setting_threshold_slider.setValue(0)
        TEST_STATUS = True

    def patience_setting(self,value):
        global TEST_STATUS
        global param
        global count
        global PATIENCE
        TEST_STATUS = False

        count = 0
        param["PATIENCE"] = int(value)
        with open('param.yaml', 'w') as file:
            yaml.dump(param, file, default_flow_style=False)
        PATIENCE = param["PATIENCE"]
        self.setting_patience_label.setText(str(PATIENCE))
        TEST_STATUS = True

    def deep_check(self):
        test.model_load()

    def module_check(self):
        global PLC_STATUS
        global PLC_BOOLEAN
        PLC_STATUS = True
        PLC_BOOLEAN = True
        self.tabWidget.setCurrentIndex(0)
        # self.real_plc = real_time_plc(self)
        # self.real_plc.start()

    def train_check(self, boolean):
        global TRAIN_STATUS
        global TEST_STATUS
        self.tabWidget.setCurrentIndex(0)
        TRAIN_STATUS = boolean
        if TRAIN_STATUS:
            TEST_STATUS = False
            self.real_test.stop()
            self.real_train.start()

        else:
            self.real_train.stop()

            TEST_STATUS = True
            self.real_test.start()
            # 파일삭제
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

        #### COMBOBOX ####
        self.setting_sec_combobox.currentIndexChanged.connect(self.sec_setting)

        #### SLIDER ####
        self.setting_threshold_slider.valueChanged[int].connect(self.threshold_setting)
        self.setting_train_slider.valueChanged[int].connect(self.train_total)
        self.setting_patience_slider.valueChanged[int].connect(self.patience_setting)

    def UIStyle(self):
        global param
        #### ICON #####
        self.tabWidget.setTabIcon(0,QIcon("./resource/pic_dashboard.png"))
        self.tabWidget.setTabIcon(1,QIcon("./resource/pic_check.png"))
        self.tabWidget.setTabIcon(2,QIcon("./resource/pic_setting.png"))
        self.close_button.setIcon(QIcon("./resource/pic_close.png"))

        #### LABEL ####
        self.time_label.setText(NOW.toString(Qt.DateFormat.ISODate))
        self.statusbar.setStyleSheet("QStatusBar{padding-left:8px;color:white}")
        self.setting_patience_label.setText(str(param["PATIENCE"]))

        #### SLIDER ####
        self.setting_patience_slider.setValue(int(param["PATIENCE"]))

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = gui()
    myWindow.show()
    app.exec()
