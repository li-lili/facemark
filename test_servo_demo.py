"""
最简Demo — 读取yaml并发送A8-A13的temp_deg值到串口
用于验证串口通信是否正常
"""

import os
import sys
from Communication import UARTDevice
from Motor import FaceController
from utility import read_yaml

# === 配置（与 run_interface.py 一致）===
PORT = "COM5"
BAUDRATE = 115200    # ★ 实际UI使用的是 115200，不是 1000000！
YAML_FILE = "29_servo_config(13).yaml"
SERVO_NUM = 27


def main():
    # 1. 初始化控制器
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), YAML_FILE)
    print(f"[1] 连接串口 {PORT} ...")
    
    interface = UARTDevice(PORT, BAUDRATE)
    controller = FaceController(interface, SERVO_NUM, yaml_path)
    controller.open()
    controller.init_data()
    print("[1] 串口连接成功!\n")
    
    # 2. 发送 A8-A13 的 temp_deg 值（逐个发送）
    eye_channels = [8, 9, 10, 11, 12, 13]
    eye_names = ["A8(左眼皮)", "A9(右眼皮)", "A10(左眼球上下)",
                 "A11(右眼球上下)", "A12(左眼球内外)", "A13(右眼球内外)"]
    
    print("[2] 发送 A8-A13 temp_deg 值 (与slider方式完全一致):")
    print("-" * 50)
    
    for ch, name in zip(eye_channels, eye_names):
        temp = controller.list_temp_deg[ch]
        start = controller.list_start_deg[ch]
        end = controller.list_end_deg[ch]
        
        print(f"  发送 {name}: angle={temp} (范围: {start}~{end})")
        
        # 与 run_interface.py scale_set_angle 完全一致
        controller.set_servo_angle_time_32(
            [temp], [ch], 200,
            servo_num=controller.servo_num
        )
    
    print("-" * 50)
    print("\n[3] 完成! 观察表情机是否有动作反应")
    
    input("\n按回车退出...")
    controller.close()
    print("已断开串口")


if __name__ == "__main__":
    main()
