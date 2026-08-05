from utility import *
from PySide6.QtWidgets import QMessageBox

class FaceController():
    def __init__(self, _interface, _servo_num:int, _yaml_file:str):
        self.interface = _interface
        if _servo_num == 17 or _servo_num == 21:
            self.servo_num = 27
        else:
            self.servo_num = _servo_num
        self.yaml_file = _yaml_file
        self.seq = 0
        self.lock = 1  # 0=锁定, 1=解锁

        self.list_index = [99] * self.servo_num
        self.list_start_deg = [0] * self.servo_num
        self.list_end_deg = [0] * self.servo_num
        self.list_temp_deg = [99] * self.servo_num

        self.valid_servo_indexes = []
        if self.servo_num == 17:
            self.valid_servo_indexes = [0,1,8,9,10,11,12,13,16,17,18,19,22,23,24,25,26]
        elif self.servo_num == 21:
            self.valid_servo_indexes = [0,1,2,3,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26]
        elif self.servo_num == 27:
            self.valid_servo_indexes = list(range(27))  # 0~26 连续

        elif self.servo_num == 29:
            self.valid_servo_indexes = list(range(29))  # 0~28 连续

    def open(self):
        try:
            return self.interface.open()
        except Exception as e:
            print(e)
            print('Error: cannot connect to serial port')
            return False
    
    def close(self):
        try:
            self.interface.close()
        except Exception as e:
            print(e)
            print('Error: cannot close serial port')

    def init_data(self):
        try:
            raw_data = read_yaml(self.yaml_file)
            
            # 适配新旧格式：优先取SERVO_INFO，无则用原始数据（旧格式）
            self.servo_name2servo_info = raw_data.get("SERVO_INFO", raw_data)  
            # 记录原文件是否为新格式（用于保存时区分）
            self.is_new_format = "SERVO_INFO" in raw_data  

            # 直接校验内存中的字典数据
            err = validate_servo_config_yaml(self.servo_name2servo_info, self.servo_num)
            if err :
                QMessageBox.critical(None, "错误", f"验证失败: {str(err)}")
                return err

            for servo_name, servo_info in self.servo_name2servo_info.items():
                if servo_info["channel_idx"] not in self.valid_servo_indexes:
                    continue
                chan_idx = servo_info["channel_idx"]
                start_deg = servo_info['start_deg']
                end_deg = servo_info['end_deg']
                temp_deg = servo_info['temp_deg']

                self.list_index[chan_idx] = chan_idx
                self.list_start_deg[chan_idx] = start_deg
                self.list_end_deg[chan_idx] = end_deg
                self.list_temp_deg[chan_idx] = temp_deg

        except Exception as e:
            print(e,111)
        print("init data completed...")

    def set_servo_angle_time_32(self, _angle, _index, _time,servo_num = None):
        cmd = self.data2cmd_32(_angle, _index, 200,servo_num)
        if not self.interface.is_open():
            print('Error: skip servo command because serial port is not open')
            return False
        self.interface.send_command(cmd)
        # 将bytes转换为十六进制字符串格式打印
        # print("send cmd: " + cmd.hex())  # 或者使用 str(cmd)
        import time
        time.sleep(0.03)
        return True
    
    def data_restart(self, _data, _index):
        angles = []
        # 处理单值情况：将 int 转换为长度为 1 的列表，统一后续处理逻辑
        if isinstance(_data, int):
            _data = [_data]
        if isinstance(_index, int):
            _index = [_index]
        
        # 校验：_data 和 _index 必须是列表且长度一致
        if not (isinstance(_data, list) and isinstance(_index, list)):
            raise TypeError("_data 和 _index 必须是 int 或 list 类型")
        if len(_data) != len(_index):
            raise ValueError(f"_data 长度（{len(_data)}）与 _index 长度（{len(_index)}）不一致")
        
        # 组合 (index, data) 元组并添加到 angles 列表
        for idx, data in zip(_index, _data):
            angles.append((idx, data))
        return angles
    
    def data2cmd_32(self, _data, _index, time_ms,servo_num):
        
        angles = self.data_restart(_data, _index)
        """
        生成舵机同步运动指令帧。

        参数:
            angles (list of tuple): [(channel_id, angle_deg), ...]
                - channel_id: int, 舵机通道 ID (0 ~ 35)
                - angle_deg: float or int, 目标角度（典型范围 0~270）
            time_ms (int): 运动时间（毫秒），0~65535
            servo_num (int): 控制舵机数量，默认 29

        返回:
            bytes: 符合协议的完整指令帧（共 129 字节）

        协议说明:
            - 帧头: 0x55 0x55
            - 长度字段 = 125（表示从 seq 到保留字段结束共 125 字节）
            - 总帧长 = 2(帧头) + 1(长度) + 125(payload) + 1(checksum) = 129 字节
            - 舵机数据固定 36 组 × 3 字节 = 108 字节
        """
        # === 输入校验 ===
        if not isinstance(angles, (list, tuple)):
            raise TypeError("angles must be a list or tuple of (channel, angle) pairs")

        if len(angles) > 36:
            raise ValueError("At most 36 servos can be controlled simultaneously")

        if not (0 <= time_ms <= 65535):
            raise ValueError("time_ms must be in range [0, 65535]")

        # 构建受控舵机字典，便于快速查找,同时进行输入校验
        servo_dict = {}
        for ch, angle in angles:
            if not isinstance(ch, int) or not (0 <= ch <= 35):
                raise ValueError(f"Channel ID must be an integer between 0 and 35, got {ch}")
            if not isinstance(angle, (int, float)):
                raise TypeError(f"Angle must be a number, got {type(angle)}")
            servo_dict[ch] = angle

        # === 开始构建命令帧 ===
        cmd = bytearray()
        cmd.extend([0x55, 0x55])          # 帧头
        cmd.append(125)                   # 长度字段：payload 长度（seq 到 reserved 共 125 字节）
        cmd.append(self.seq)
        self.seq = (self.seq + 1) % 256   # 序列号递增并回绕
        cmd.append(self.lock)             # 锁标志
        cmd.append(0x03)                  # 指令类型：SERVO_MOVE

        if servo_num == 21:
            servo_num = 27
        elif servo_num == 17:
            servo_num=27
        cmd.append(servo_num)           # 实际控制的舵机数量

        # 时间（高8位 + 低8位）
        cmd.append((time_ms >> 8) & 0xFF)
        cmd.append(time_ms & 0xFF)

        # 舵机数据：固定 36 组（ID 0~35）
        for ch in range(36):
            if ch in servo_dict:
                angle = servo_dict[ch]
                # 角度转脉宽：0° → 500μs, 270° → 2500μs => scale = 2000/270 ≈ 7.407
                pulse = 500 + round(angle * 7.407)
                pulse = max(500, min(pulse, 2500))  # 限幅到合法范围
            else:
                pulse = 0  # 未控制的舵机设为 0（通常表示不动作）

            cmd.append(ch)
            cmd.append((pulse >> 8) & 0xFF)
            cmd.append(pulse & 0xFF)

        # 保留字段（10 字节）
        cmd.extend([0x00] * 10)

        # 校验和：所有已写入字节之和 mod 256
        checksum = sum(cmd) & 0xFF
        cmd.append(checksum)

        return bytes(cmd)
