import serial
import serial.tools.list_ports as sp

import platform

connection_code = '\x0501RSS0106%MW003\x04'.encode()
error_code = '\x0501WSS0106%MW0030001\x04'.encode()
output = [False, None]

## output : [connection_success , platform, port_name]
def port_init():
    list_port = sp.comports()
    maybe_port = []
    try:
        for i in list_port:
            port_temp = str(i).split(" ")[0]
            if "usbserial" or "COM" in port_temp:
                maybe_port.append(port_temp)
            else:
                output = [False, None]
        for port in maybe_port:
            ser = serial.Serial(port = port, baudrate = 9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS, timeout=1)
            ser.write(connection_code)
            result = ser.readline().decode('ascii')

            if '\x06' in result:
                output = [True, port]
                break
                    
        return output

    except:
        output = [False, None]
        return output


def port_error_message(port):
    ser = serial.Serial(port = port, baudrate = 9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS, timeout=1)
    try:
        ser.write(error_code)
        result = ser.readline().decode('ascii')
        if '\x06' in result:
            return True
        else:
            return False
    except:
        return False
