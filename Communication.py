import serial


class UARTDevice:
    def __init__(self, port, baudrate, auto_open=False):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        if auto_open:
            self.open()

    def is_open(self):
        return bool(self.ser and self.ser.is_open)
        
    def open(self):
        if self.is_open():
            return True
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            return True
        except Exception as e:
            print(e)
            print('Error: cannot connect to serial port')
            self.ser = None
            return False
    
    def close(self):
        if self.ser:
            try:
                self.ser.close()
            finally:
                self.ser = None
    
    def send_command(self, command):
        if self.is_open():
            self.ser.write(command)
        # return self.ser.read(32)
    
    def receive_data(self):
        return self.ser.readline()
