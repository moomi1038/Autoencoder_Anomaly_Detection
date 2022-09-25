import serial
import serial.tools.list_ports as sp

import platform

plf = platform.system()

connection_code = str('\x0501RSS0106%MW' + '003' +'\x04').encode()
error_code = str('\x0501WSS0106%MW' + '003' + '0001' + '\x04').encode()

output = [False, plf, " "]

## output : [connection_success , platform, port_name]
def port_init():
    global plf
    global connection_code
    global output
    
    list_port = sp.comports()
    for i in list_port:
        port_temp = str(i).split(" ")[0]
        if "usbserial" or "COM" in port_temp:
            ser = serial.Serial(port = port_temp, baudrate = 9600, timeout = 1)
            try:
                ser.write(connection_code)
                result = ser.readline().decode('ascii')
                if result[0] == '\x06':
                    output[0] = True
                    output[1] = plf
                    output[2] = port_temp
                    return output
                    break
            except:
                continue

    ### For memory leak
    del port_temp
    del result 
    del list_port

def port_error_message():
    global output
    global error_code
    port = output[2]
    ser = serial.Serial(port = port, baudrate = 9600, timeout = 1)
    try:
        ser.write(error_code)
        result = ser.readline().decode('ascii')
        if result[0] == '\x06':
            return True
        else:
            return False
    except:
        return False