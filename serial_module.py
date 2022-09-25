import serial
import serial.tools.list_ports as sp

import platform

## output : [connection_success , port_name]
def port_init():
    list_port = sp.comports()
    plf = platform.system()
    

def port_error_message():
    
port_init()