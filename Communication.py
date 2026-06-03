import serial


class UARTDevice:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.open()
        
    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception as e:
            print(e)
            print('Error: cannot connect to serial port')
    
    def close(self):
        if self.ser:
            self.ser.close()
    
    def send_command(self, command):
        if self.ser:
            self.ser.write(command)
        # return self.ser.read(32)
    
    def receive_data(self):
        return self.ser.readline()