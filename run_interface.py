import os
import sys
import copy
import glob 
import yaml
from functools import partial
import resources_rc
import serial  
from PySide6.QtWidgets import (
                QApplication, QWidget, QMessageBox, QFileDialog,QComboBox,
                QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
                )
from PySide6.QtCore import Slot, QDir,Qt, Signal
from UI.interface_ui_17 import Ui_Form as ui_17  # 27舵机UI
from UI.interface_ui_21 import Ui_Form as ui_21  # 29舵机UI
from UI.interface_ui_27 import Ui_Form as ui_27  # 27舵机UI
from UI.interface_ui_29 import Ui_Form as ui_29  # 29舵机UI
from UI.login_ui import *
from Motor import FaceController
from Communication import UARTDevice
from utility import *

class CustomConfirmDialog(QDialog):
    """自定义确认对话框 - 覆盖/另存为"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("保存配置")
        self.setModal(True)  # 模态对话框，阻塞父窗口
        self.setFixedSize(350, 150)
        # 布局
        v_layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        # 提示文本
        label = QLabel("是否覆盖原有配置文件？")
        label.setAlignment(Qt.AlignCenter)
        v_layout.addWidget(label)
        # 按钮
        self.btn_overwrite = QPushButton("确定（覆盖）")
        self.btn_saveas = QPushButton("另存为")
        self.btn_cancel = QPushButton("取消")
        
        h_layout.addWidget(self.btn_overwrite)
        h_layout.addWidget(self.btn_saveas)
        h_layout.addWidget(self.btn_cancel)

        v_layout.addLayout(h_layout)
        self.setLayout(v_layout)
        # 绑定按钮事件
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_overwrite.clicked.connect(self.accept_overwrite)
        self.btn_saveas.clicked.connect(self.accept_saveas)
        # 标记用户选择
        self.choice = None 

    def accept_overwrite(self):
        self.choice = 'overwrite'
        self.accept()

    def accept_saveas(self):
        self.choice = 'saveas'
        self.accept()

class FaceCalibrationWidget(QWidget):
    def __init__(self, parent = None, serial_port = None, config_fn = None,servo_num = None):
        super().__init__(parent)

        # 1. 创建登录页
        login_widget = QWidget()
        login_ui = Ui_Form()
        login_ui.setupUi(login_widget)
        login_widget.show()
        # 直接在这里绑定，不去碰UI文件！
        login_ui.pushButton.clicked.connect(lambda: self.on_select_servo_num(29))
        login_ui.pushButton_2.clicked.connect(lambda: self.on_select_servo_num(27))
        login_ui.pushButton.clicked.connect(lambda: self.on_select_servo_num(21))
        login_ui.pushButton_2.clicked.connect(lambda: self.on_select_servo_num(17))
        
        # ===== 核心：按你给的索引定义有效滑块列表 =====
        self.servo_num = servo_num  # 舵机数量赋值
        self.valid_servo_indexes = []
        if self.servo_num == 17:
            self.ui = ui_17()  # 你自己的17舵机UI类名
            # 你提供的 17 个有效索引
            self.valid_servo_indexes = [0,1,8,9,10,11,12,13,16,17,18,19,22,23,24,25,26]
        elif self.servo_num == 21:
            self.ui = ui_21()  # 你自己的21舵机UI类名
            # 你提供的 21 个有效索引
            self.valid_servo_indexes = [0,1,2,3,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26]
        elif self.servo_num == 27:
            self.ui = ui_27()
            self.valid_servo_indexes = list(range(27))  # 0~26 连续

        elif self.servo_num == 29:
            self.ui = ui_29()
            self.valid_servo_indexes = list(range(29))  # 0~28 连续
        else:
            raise ValueError("仅支持17/21/27/29舵机数量")
        self.ui.setupUi(self)  # 初始化UI
        self.uart_device = None

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.serial_port = serial_port
        self.config_fn = config_fn
        
        # 核心修改3：适配is_default_direction_dict的长度（如果27/29的字典不同，需分别定义）
        # 基础27舵机方向配置（17/21复用此配置，29补充A27/A28）
        base_direction_dict = {
            'A0': True, 'A1': True, 'A2': True, 'A3': False, 'A4': False, 'A5': False, 
            'A6': False, 'A7': False, 'A8': True, 'A9': True, 'A10': False, 'A11': True, 
            'A12': True, 'A13': True, 'A14': False, 'A15': False, 'A16': False, 'A17': False, 
            'A18': False, 'A19': False, 'A20': False, 'A21': False, 'A22': False, 'A23': True, 
            'A24': False, 'A25': False, 'A26': False
        }
        # 按舵机数量初始化方向字典
        if self.servo_num in [17, 21, 27]:
            self.is_default_direction_dict = base_direction_dict
        elif self.servo_num == 29:
            # 29舵机补充A27/A28的方向配置
            self.is_default_direction_dict = base_direction_dict.copy()
            self.is_default_direction_dict.update({'A27': False, 'A28': False})
        
        self.folder_path = os.getcwd()
        self.neutral_degree = [135] * 29

        # ========== 新增：串口初始化相关 ==========
        self.init_serial_combo()  # 初始化串口下拉框
        self.scan_serial_ports()  # 自动扫描可用串口

        # 新增串口状态变量
        self.serial_opened = False  # 串口是否已打开
        self.config_loaded = False   # 配置文件是否已载入

        self.init_Slider()

        # 新增：初始化时禁用所有控制组件 + 绑定打开串口按钮事件
        self.set_control_widgets_state(False)
        self.ui.pushButton_load.setEnabled(False)
        self.bind_serial_button_event()

    def on_refresh_serial_clicked(self):
        """刷新串口按钮点击事件处理"""
        # 1. 检查串口是否已打开，若打开则提示并返回
        if self.serial_opened:
            QMessageBox.warning(self, "提示", "请先关闭串口后再刷新可用串口！")
            return
        # 2. 执行串口扫描并更新下拉框
        try:
            self.scan_serial_ports()  # 复用原有串口扫描逻辑
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新串口失败：{str(e)}")

    def bind_serial_button_event(self):
        """绑定打开串口按钮的点击事件（先确认UI里按钮的objectName，假设是 pushButton_openORclose_serial）"""
        # 替换为你实际的打开串口按钮objectName（从interface_ui.py里找）
        self.ui.pushButton_openORclose_serial.clicked.connect(self.on_openORclose_serial_clicked)

    def on_openORclose_serial_clicked(self):
        """打开/关闭串口按钮点击事件"""
        # 1. 串口未打开：执行打开逻辑
        if not self.serial_opened:
            if not self.serial_port or self.serial_port == "无可用串口":
                QMessageBox.warning(self, "提示", "请先选择可用串口！")
                return
            try:
                # 创建UARTDevice实例并打开串口
                self.uart_device = UARTDevice(self.serial_port, 115200, auto_open=True)
                # 校验串口是否真的打开（检查serial.Serial实例）
                if self.uart_device.ser and self.uart_device.ser.is_open:
                    self.serial_opened = True
                    # 修改按钮文字+样式（浅红色）
                    self.ui.pushButton_openORclose_serial.setText("关闭串口")
                    self.ui.pushButton_openORclose_serial.setStyleSheet("""
                        QPushButton {
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                                        stop:0 #fff0ee, stop:0.5 #ffe8e6, stop:1 #ffdedb);
                            color: #cf1322;
                            font-weight: bold;
                            border-color: #ffccc7 #ff9a9e #ff9a9e #ffccc7;
                            border-radius: 6px;
                            padding: 6px 16px;
                            min-height: 28px;
                            min-width: 80px;
                            font-size: 9pt;
                        }
                    """)
                    # ========== 新增/修改：串口打开后逻辑 ==========
                    self.ui.pushButton_load.setEnabled(True)  # 启用载入配置按钮
                    # 如果配置已载入，直接启用控制组件
                    if self.config_loaded:
                        self.set_control_widgets_state(True)
                        # 可选：重新初始化face_controller，确保串口实例最新
                        if self.config_fn and os.path.exists(self.config_fn):
                            self.init_face_controller()
                            
                    self.reset_all_servos_to_initial()
                else:
                    QMessageBox.critical(self, "失败", "串口实例创建失败！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"打开串口失败：{str(e)}")
        # 2. 串口已打开：执行关闭逻辑
        else:
            try:
                self.uart_device.close()
                self.serial_opened = False
                # 恢复按钮文字+样式
                self.ui.pushButton_openORclose_serial.setText("打开串口")
                self.ui.pushButton_openORclose_serial.setStyleSheet("")  # 恢复默认样式
                # 关闭串口后禁用控制组件
                self.set_control_widgets_state(False)
                # ========== 新增：串口关闭后禁用载入配置按钮 ==========
                self.ui.pushButton_load.setEnabled(False)
                # ===================================================
            except Exception as e:
                QMessageBox.critical(self, "错误", f"关闭串口失败：{str(e)}")

    def set_control_widgets_state(self, enabled):
        """设置控制模块所有组件的启用/禁用状态"""
        # 示例：禁用/启用所有Slider、SpinBox、功能按钮（根据你的UI组件补充）
        for i in self.valid_servo_indexes:
            # 启用/禁用Slider
            slider = getattr(self.ui, f"Slider_{i}", None)
            if slider:
                slider.setEnabled(enabled)
            # 启用/禁用SpinBox（M/S/E）
            spin_m = getattr(self.ui, f"spinBox_{i}_M", None)
            spin_s = getattr(self.ui, f"spinBox_{i}_S", None)
            spin_e = getattr(self.ui, f"spinBox_{i}_E", None)
            if spin_m: spin_m.setEnabled(enabled)
            if spin_s: spin_s.setEnabled(enabled)
            if spin_e: spin_e.setEnabled(enabled)
        
        # 启用/禁用其他功能按钮（保存、重置等）
        self.ui.pushButton_save.setEnabled(enabled)
        self.ui.pushButton_reset_state.setEnabled(enabled)
        self.ui.pushButton_refresh.setEnabled(enabled)
        self.ui.pushButton_load_state.setEnabled(enabled)
        self.ui.pushButton_save_state.setEnabled(enabled)
        # 补充你实际的控制组件...
    
    def __del__(self):
        """窗口销毁时释放串口"""
        if hasattr(self, 'face_controller'):
            try:
                self.face_controller.close()
            except:
                pass
        # 新增：关闭串口并重置状态
        if self.serial_opened and self.uart_device:
            self.uart_device.close()
            self.serial_opened = False

    def close(self):
        """关闭串口方法"""
        if hasattr(self, 'uart_device'):
            self.uart_device.close()  # 调用 UARTDevice 的关闭方法
    
     # ========== 新增：串口扫描（兼容Windows/Linux/Mac） ==========
    def scan_serial_ports(self):
        """自动扫描可用串口并更新到下拉框"""
        self.ui.comboBox_serial_port.clear()  # 清空原有选项
        ports = []
        try:
            if sys.platform.startswith('win'):  # Windows
                ports = [f'COM{i}' for i in range(1, 20)]
            elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):  # Linux
                ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            elif sys.platform.startswith('darwin'):  # Mac
                ports = glob.glob('/dev/tty.usbserial*') + glob.glob('/dev/tty.usbmodem*')
            # 验证串口是否可用
            available_ports = []
            for port in ports:
                try:
                    s = serial.Serial(port)
                    s.close()
                    available_ports.append(port)
                except (OSError, serial.SerialException):
                    pass
            if available_ports:
                self.ui.comboBox_serial_port.addItems(available_ports)
                # 若初始化传入了串口，自动选中
                if self.serial_port and self.serial_port in available_ports:
                    self.ui.comboBox_serial_port.setCurrentText(self.serial_port)
            else:
                self.ui.comboBox_serial_port.addItem("无可用串口")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"串口扫描失败：{str(e)}")
    # ============================================================

    # ========== 新增：初始化串口下拉框（绑定信号） ==========
    def init_serial_combo(self):
        """初始化串口下拉框的交互：选择/双击"""
        # 下拉框选择事件
        self.ui.comboBox_serial_port.currentTextChanged.connect(self.on_serial_port_selected)
        # 双击下拉框选项选中
        self.ui.comboBox_serial_port.view().doubleClicked.connect(self.on_serial_port_double_clicked)

    # ========== 新增/修改：串口选择相关槽函数 ==========
    def on_serial_port_selected(self, port):
        """串口下拉框选择事件"""
        if port != "无可用串口":
            self.serial_port = port

    def on_serial_port_double_clicked(self, index):
        """双击下拉框选项选中串口"""
        port = self.ui.comboBox_serial_port.itemText(index.row())
        if port != "无可用串口":
            self.serial_port = port
            QMessageBox.information(self, "提示", f"已选中串口：{port}")

    def reset_all_servos_to_initial(self):
        """只有已载入配置时，才复位所有舵机到初始位置"""
        # 关键：没有载入配置 → 直接退出，不发任何指令
        if not self.config_loaded:
            return
        # 串口没打开也不执行
        if not self.uart_device or not self.serial_opened:
            return
        try:
            # 遍历所有舵机，发送当前保存的初始角度
            for index in range(self.servo_num):
                initial_deg = self.neutral_degree[index]
                # 发送复位指令
                self.scale_set_angle(initial_deg, f"A{index}", index)
                # 同步UI
                slider = getattr(self.ui, f"Slider_{index}", None)
                spinbox_m = getattr(self.ui, f"spinBox_{index}_M", None)
                if slider and spinbox_m:
                    slider.setValue(initial_deg)
                    spinbox_m.setValue(initial_deg)
        except Exception as e:
            print(f"舵机复位异常：{e}")

    def init_face_controller(self):
        # 使用验证过的串口实例
        self.face_controller = FaceController(
            self.uart_device, 
            self.servo_num,
            self.config_fn)
        # ========== 新增：缓存原文件的HW_INFO ==========
        self.origin_hw_info = read_yaml(self.config_fn).get("HW_INFO", {}) if self.config_fn else {}  
        # ========== 新增：角度范围校验 + 异常提示 ==========
        err = self.face_controller.init_data()
        if err:
            return err
        # 4. 先断开所有滑块的信号连接，避免更新UI时发送指令
        self.angles = []  # 存储所有舵机的目标角度和索引
        # 关键修改1：仅遍历当前舵机数量的有效索引，而非全量
        for i in self.valid_servo_indexes:
            slider = getattr(self.ui, f"Slider_{i}", None)
            if slider:  # 仅当控件存在时才断开信号
                slider.valueChanged.disconnect()
        # 5. 初始化滑块和SpinBox
        # 关键修改2：过滤出当前有效索引的舵机信息
        for servo_name, servo_info in self.face_controller.servo_name2servo_info.items():
            channel_idx = servo_info['channel_idx']
            # 仅处理当前舵机数量的有效索引
            if channel_idx not in self.valid_servo_indexes:
                continue
            
            # 安全获取控件（避免访问不存在的控件）
            slider = getattr(self.ui, f"Slider_{channel_idx}", None)
            spinbox_m = getattr(self.ui, f"spinBox_{channel_idx}_M", None)
            spinbox_s = getattr(self.ui, f"spinBox_{channel_idx}_S", None)
            spinbox_e = getattr(self.ui, f"spinBox_{channel_idx}_E", None)
            
            # 仅当所有控件存在时才初始化
            if not all([slider, spinbox_m, spinbox_s, spinbox_e]):
                continue
            
            # 设置start/end并更新滑块范围
            spinbox_s.setValue(servo_info['start_deg'])
            spinbox_e.setValue(servo_info['end_deg'])
            self.on_spinbox_s_editing_finished(channel_idx)
            temp_deg = servo_info['temp_deg']
            
            # 设置调整后的temp_deg到UI
            spinbox_m.setValue(temp_deg)
            slider.setValue(temp_deg)
            self.neutral_degree[channel_idx] = temp_deg
            
            # 收集目标角度
            self.angles.append((channel_idx, temp_deg))
        
        # 6. 重新连接所有滑块的信号
        # 关键修改3：仅重新连接有效索引的滑块
        for i in self.valid_servo_indexes:
            slider = getattr(self.ui, f"Slider_{i}", None)
            if slider:  # 仅当控件存在时才连接信号
                slider.valueChanged.connect(partial(self.on_slider_value_changed, i))
        
        # 7. 发送一条批量指令
        self.many_angles()

    def many_angles(self):
        if not self.angles:
            return
        # 每组最多36个舵机指令
        batch_size = 36
        # 按36个为一组拆分angles列表
        for i in range(0, len(self.angles), batch_size):
            batch = self.angles[i:i+batch_size]
            # 提取当前组的角度和索引
            angles_batch = [angle for _, angle in batch]
            idx_batch = [idx for idx, _ in batch]
            # 发送当前组的批量指令
            self.face_controller.set_servo_angle_time_32(
                angles_batch,
                idx_batch,
                200,
                servo_num = self.servo_num
            )
    
    def scale_set_angle(self, _angle, _name, _index):
        self.face_controller.set_servo_angle_time_32([_angle], [_index], 200,servo_num = self.servo_num)

    @Slot()
    def on_pushButton_load_clicked(self):
        # 载入配置文件
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择配置文件",
            self.folder_path
        )
        if file_path:
            self.config_fn = file_path
            if not os.path.exists(self.config_fn):
                QMessageBox.critical(self, "错误", "文件不存在")
                return
            if self.init_face_controller():
                return
            # 1. 获取配置文件所在目录
            config_dir = os.path.dirname(self.config_fn)
            # 2. 显示配置所在的最后一级文件夹名 → lineEdit_config
            last_folder_name = os.path.basename(config_dir)
            self.ui.lineEdit_config.setText(last_folder_name)
            #3. 自动创建/使用 bs 文件夹
            bs_folder = os.path.join(config_dir, "bs")
            if not os.path.exists(bs_folder):
                os.makedirs(bs_folder)
            # 4. 显示：bs文件夹 + 上一级文件夹 → lineEdit_bs
            parent_folder = os.path.basename(config_dir)  # 上一级文件夹名
            bs_display = f"{parent_folder}/bs"             # 显示格式：文件夹/bs
            self.ui.lineEdit_bs.setText(bs_display)
            # 5. 关键：自动更新表情文件夹列表（解决不刷新问题）
            self.folder_path = bs_folder
            self.load_yaml_files_in_folder()  # 强制刷新表情文件
            self.config_loaded = True
            if self.serial_opened:
                self.set_control_widgets_state(True)
    
    def _get_save_config_data(self):
        """提取当前页面配置数据（复用逻辑）"""
        save_config = copy.deepcopy(self.face_controller.servo_name2servo_info)
        # 仅遍历有效索引，过滤不存在的舵机
        for name, info in list(save_config.items()):
            idx = info['channel_idx']
            if idx not in self.valid_servo_indexes:
                continue  # 跳过无效索引
            # 安全获取控件，不存在则跳过
            sld = getattr(self.ui, f"Slider_{idx}", None)
            sm = getattr(self.ui, f"spinBox_{idx}_M", None)
            ss = getattr(self.ui, f"spinBox_{idx}_S", None)
            se = getattr(self.ui, f"spinBox_{idx}_E", None)
            if not all([sld, sm, ss, se]):
                continue
            info['start_deg'] = ss.value()
            info['end_deg'] = se.value()
            info['temp_deg'] = sld.value()

        if hasattr(self.face_controller, 'is_new_format') and self.face_controller.is_new_format:
            return {"HW_INFO": self.origin_hw_info, "SERVO_INFO": save_config}
        else:
            return save_config

    def _update_page_config(self):
        """更新页面数据为最新保存的配置"""
        if not self.config_fn or not os.path.exists(self.config_fn):
            return
        # 重新初始化控制器，刷新页面控件值
        self.init_face_controller()

    @Slot()
    def on_pushButton_save_clicked(self):
        # 1. 检查是否已载入配置文件（无则直接走另存为）
        if not self.config_fn or not os.path.exists(self.config_fn):
            self._handle_save_as()
            return
        # 2. 弹出自定义确认对话框
        dialog = CustomConfirmDialog(self)
        result = dialog.exec()
        if result == QDialog.Rejected:
            return  # 用户取消
        # 3. 根据用户选择处理
        if dialog.choice == 'overwrite':
            self._handle_overwrite()
        elif dialog.choice == 'saveas':
            self._handle_save_as()

    def _handle_overwrite(self):
        """处理覆盖保存"""
        try:
            # 提取配置数据
            save_config = self._get_save_config_data()
            # 覆盖原文件
            with open(self.config_fn, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            # 更新页面数据
            self._update_page_config()
            QMessageBox.information(self, "提示", f"配置已覆盖保存！\n路径: {self.config_fn}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"覆盖保存失败: {str(e)}")

    def _handle_save_as(self):
        """处理另存为（定位到载入配置的文件夹）"""
        # 确定默认路径：优先使用当前载入配置的文件夹，其次用当前工作目录
        default_dir = os.path.dirname(self.config_fn) if self.config_fn else self.folder_path
        default_filename = os.path.basename(self.config_fn) if self.config_fn else "new_config.yaml"
     
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "另存为配置文件",
            os.path.join(default_dir, default_filename),
            "YAML 文件 (*.yaml *.yml);;所有文件 (*)"
        )
        if not file_path:
            return 
        # 确保扩展名正确
        if not file_path.lower().endswith(('.yaml', '.yml')):
            file_path += '.yaml'
        try:
            save_config = self._get_save_config_data()
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            QMessageBox.information(self, "提示", f"配置已另存为！\n路径: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"另存为失败: {str(e)}")

    @Slot()
    def on_pushButton_refresh_clicked(self):
        self.load_yaml_files_in_folder()
    
    def load_yaml_files_in_folder(self):
        """加载并显示 YAML 文件"""
        self.ui.listWidget_file.clear()
        if not os.path.isdir(self.folder_path):
            self.ui.listWidget_file.addItem(f"无效的文件夹路径: {self.folder_path}")
            return
        # 获取所有 .yaml 文件
        dir_obj = QDir(self.folder_path)
        yaml_files = dir_obj.entryList(["*.yaml", "*.yml"], QDir.Files)
        if not yaml_files:
            self.ui.listWidget_file.addItem("该文件夹中没有找到 YAML 文件")
            return
        # 添加到列表控件
        self.ui.listWidget_file.addItems(yaml_files)
        # 设置双击打开文件的功能
        self.ui.listWidget_file.itemDoubleClicked.connect(self.on_pushButton_load_state_clicked)
    
    @Slot()
    def on_pushButton_load_state_clicked(self):
        items = self.ui.listWidget_file.selectedItems()
        if not items:
            return
        fn = items[0].text()
        path = os.path.join(self.folder_path, fn)
        try:
            data = yaml.safe_load(open(path, 'r', encoding='utf8'))
            # 仅处理有效索引，跳过无效索引
            for i in self.valid_servo_indexes:
                if i >= len(data):
                    continue  # 跳过超出数据长度的索引
                s = getattr(self.ui, f"Slider_{i}", None)
                if s:
                    try:
                        s.valueChanged.disconnect()
                    except:
                        pass

            self.angles = []
            for i in self.valid_servo_indexes:
                if i >= len(data):
                    continue
                target = data[i]
                s = getattr(self.ui, f"Slider_{i}", None)
                sm = getattr(self.ui, f"spinBox_{i}_M", None)
                ss = getattr(self.ui, f"spinBox_{i}_S", None)
                se = getattr(self.ui, f"spinBox_{i}_E", None)
                if not all([s,sm,ss,se]):
                    continue
                fixed = self.clamp_angle(target, ss.value(), se.value())
                s.setValue(fixed)
                sm.setValue(fixed)
                self.angles.append((i, fixed))

            for i in self.valid_servo_indexes:
                s = getattr(self.ui, f"Slider_{i}", None)
                if s:
                    s.valueChanged.connect(partial(self.on_slider_value_changed, i))

            self.many_angles()
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    # @Slot()
    # def on_pushButton_load_state_clicked(self):
    #     """载入选中的 YAML 文件"""
    #     selected_items = self.ui.listWidget_file.selectedItems()
    #     if not selected_items:
    #         QMessageBox.warning(self, "警告", "请先选择一个 YAML 文件")
    #         return

    #     file_name = selected_items[0].text()
    #     full_path = os.path.join(self.folder_path, file_name)
    #     try:
    #         with open(full_path, 'r', encoding='utf-8') as f:
    #             current_state = yaml.safe_load(f) or []
    #             if len(current_state) != self.servo_num:
    #                 QMessageBox.critical(self, "错误", f"文件格式错误，需要包含{self.servo_num}个舵机状态")
    #                 return
    #         # 1. 先断开所有滑块的信号连接，避免更新UI时发送指令
    #         for index in range(self.servo_num):
    #             slider = getattr(self.ui, f"Slider_{index}")
    #             slider.valueChanged.disconnect()
    #         # 2. 收集角度有变化的舵机，同时校验并修正角度范围
    #         self.angles = []  # 只存变化的舵机 (索引, 目标角度)
    #         error_info = []   # 记录超出范围的舵机信息
    #         adjusted_info = []# 记录自动调整的舵机信息
    #         for index in range(self.servo_num):
    #             # 获取当前UI角度、目标角度、start/end范围
    #             current_angle = getattr(self.ui, f"Slider_{index}").value()
    #             target_angle = current_state[index]
    #             spinbox_s = getattr(self.ui, f"spinBox_{index}_S")
    #             spinbox_e = getattr(self.ui, f"spinBox_{index}_E")
    #             start_deg = spinbox_s.value()
    #             end_deg = spinbox_e.value()
    #             # 校验并修正目标角度到start/end范围内
    #             clamped_angle = self.clamp_angle(target_angle, start_deg, end_deg)
    #             # 记录超出范围的情况
    #             if clamped_angle != target_angle:
    #                 target_angle = clamped_angle  # 使用修正后的角度
    #             # 仅当角度变化时更新UI并收集
    #             if current_angle != target_angle:
    #                 # 更新滑块值
    #                 slider = getattr(self.ui, f"Slider_{index}")
    #                 slider.setValue(target_angle)
    #                 # 同步更新对应的SpinBox
    #                 spinbox_m = getattr(self.ui, f"spinBox_{index}_M")
    #                 spinbox_m.setValue(target_angle)   
    #                 # 收集变化的舵机信息
    #                 self.angles.append((index, target_angle))
    #         # 3. 重新连接所有滑块的信号
    #         for index in range(self.servo_num):
    #             slider = getattr(self.ui, f"Slider_{index}")
    #             slider.valueChanged.connect(partial(self.on_slider_value_changed, index))
    #         # 4. 仅发送变化的舵机指令（无变化则不发送）
    #         if self.angles:
    #             print(f"载入状态：仅发送 {len(self.angles)} 个变化舵机的指令")
    #             self.many_angles()  
    #         # 5. 弹窗提示角度异常/调整信息
    #         if error_info:
    #             tip_text = "以下舵机载入角度超出当前start/end范围：\n" + "\n".join(error_info)
    #             if adjusted_info:
    #                 tip_text += "\n\n已自动调整为合法范围：\n" + "\n".join(adjusted_info)
    #             QMessageBox.warning(self, "载入动作角度校验提示", tip_text)
                    
        except Exception as e:
            QMessageBox.critical(self, "错误", f"载入文件失败: {str(e)}")
            # 发生错误时也需要重新连接信号，避免UI交互失效
            for index in range(self.servo_num):
                slider = getattr(self.ui, f"Slider_{index}")
                try:
                    slider.valueChanged.connect(partial(self.on_slider_value_changed, index))
                except:
                    pass
    
    def clamp_angle(self, angle, start_deg, end_deg):
        """将角度限制在start和end范围内（自动适配大小）"""
        min_deg = min(start_deg, end_deg)
        max_deg = max(start_deg, end_deg)
        # 确保角度在[min_deg, max_deg]之间
        clamped_angle = max(min(angle, max_deg), min_deg)
        return clamped_angle

    @Slot()
    def on_pushButton_reset_state_clicked(self):
        """复原初始动作"""
        try:
            self.angles = []
            changed = []
            # 仅遍历有效索引
            for i in self.valid_servo_indexes:
                sld = getattr(self.ui, f"Slider_{i}", None)
                smm = getattr(self.ui, f"spinBox_{i}_M", None)
                sss = getattr(self.ui, f"spinBox_{i}_S", None)
                see = getattr(self.ui, f"spinBox_{i}_E", None)
                if not all([sld, smm, sss, see]):
                    continue
                cur = sld.value()
                target = self.clamp_angle(self.neutral_degree[i], sss.value(), see.value())
                if cur != target:
                    changed.append(i)
                    self.angles.append((i, target))

            # 仅断开有效索引的信号
            for i in changed:
                if i not in self.valid_servo_indexes:
                    continue
                s = getattr(self.ui, f"Slider_{i}", None)
                if s:
                    try:
                        s.valueChanged.disconnect()
                    except:
                        pass

            # 仅更新有效索引的UI
            for i in changed:
                if i not in self.valid_servo_indexes:
                    continue
                target = self.clamp_angle(self.neutral_degree[i],
                    getattr(self.ui, f"spinBox_{i}_S").value(),
                    getattr(self.ui, f"spinBox_{i}_E").value())
                s = getattr(self.ui, f"Slider_{i}", None)
                m = getattr(self.ui, f"spinBox_{i}_M", None)
                if s and m:
                    s.setValue(target)
                    m.setValue(target)

            # 仅重连有效索引的信号
            for i in changed:
                if i not in self.valid_servo_indexes:
                    continue
                s = getattr(self.ui, f"Slider_{i}", None)
                if s:
                    s.valueChanged.connect(partial(self.on_slider_value_changed, i))

            self.many_angles()
        except Exception as e:
            QMessageBox.critical(self, "失败", str(e))


    @Slot()
    def on_pushButton_save_state_clicked(self):
        self.save_current_state()

    def save_current_state(self):
        """保存当前状态到yaml文件"""
        path, _ = QFileDialog.getSaveFileName(self, "保存状态", self.folder_path, "*.yaml *.yml")
        if not path:
            return
        if not path.lower().endswith(('.yaml','.yml')):
            path += '.yaml'
        try:
            # 初始化长度29的列表，兼容最大索引26
            out = [0]*29
            # 仅写入有效索引的角度
            for i in self.valid_servo_indexes:
                s = getattr(self.ui, f"Slider_{i}", None)
                if s:
                    out[i] = s.value()
            with open(path, 'w', encoding='utf8') as f:
                yaml.dump(out, f, allow_unicode=True)
            QMessageBox.information(self, "成功", "保存完成")
            self.load_yaml_files_in_folder()
        except Exception as e:
            QMessageBox.critical(self, "失败", str(e))
    

    def init_Slider(self):
        # 只遍历你指定的有效索引，不存在的自动跳过
        for i in self.valid_servo_indexes:

            # 安全获取组件：不存在就返回 None，不崩溃
            slider = getattr(self.ui, f"Slider_{i}", None)
            spinbox_m = getattr(self.ui, f"spinBox_{i}_M", None)
            spinbox_s = getattr(self.ui, f"spinBox_{i}_S", None)
            spinbox_e = getattr(self.ui, f"spinBox_{i}_E", None)

            # 只有组件存在时才绑定
            if all([slider, spinbox_m, spinbox_s, spinbox_e]):
                slider.valueChanged.connect(partial(self.on_slider_value_changed, i))
                spinbox_m.editingFinished.connect(partial(self.on_spinbox_m_editing_finished, i))
                spinbox_s.editingFinished.connect(partial(self.on_spinbox_s_editing_finished, i))
                spinbox_e.editingFinished.connect(partial(self.on_spinbox_e_editing_finished, i))

    def on_slider_value_changed(self, index, value):
        slider = getattr(self.ui, f"Slider_{index}")
        spinbox_m = getattr(self.ui, f"spinBox_{index}_M")
        spinbox_m.setValue(slider.value())
        self.scale_set_angle(slider.value(), f"A{index}", index)

    def on_spinbox_m_editing_finished(self, index):
        spinbox_m = getattr(self.ui, f"spinBox_{index}_M")
        slider = getattr(self.ui, f"Slider_{index}")
        slider.setValue(spinbox_m.value())

    def on_spinbox_s_editing_finished(self, index):
        spinbox_s = getattr(self.ui, f"spinBox_{index}_S")
        spinbox_e = getattr(self.ui, f"spinBox_{index}_E")
        slider = getattr(self.ui, f"Slider_{index}")
        # 容错：如果索引不在方向字典中，默认使用True
        dir_key = f"A{index}"
        is_default = self.is_default_direction_dict.get(dir_key, True)
        if is_default:
            if spinbox_s.value() < spinbox_e.value():
                slider.setMaximum(spinbox_e.value())
                slider.setMinimum(spinbox_s.value())
                slider.setInvertedAppearance(False)
            else:
                slider.setMaximum(spinbox_s.value())
                slider.setMinimum(spinbox_e.value())
                slider.setInvertedAppearance(True)
        else:
            if spinbox_s.value() < spinbox_e.value():
                slider.setMinimum(spinbox_s.value())
                slider.setMaximum(spinbox_e.value())
                slider.setInvertedAppearance(True)
            else:
                slider.setMinimum(spinbox_e.value())
                slider.setMaximum(spinbox_s.value())
                slider.setInvertedAppearance(False)

    def on_spinbox_e_editing_finished(self, index):
        self.on_spinbox_s_editing_finished(index)

if __name__ == "__main__":
    import sys
    from UI.login_ui import Ui_Form

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 1. 创建登录页
    login_widget = QWidget()
    login_ui = Ui_Form()
    login_ui.setupUi(login_widget)
    login_widget.show()

    # 2. 点击按钮跳转（这里自动匹配你UI里的按钮名，不会报错）
    def go_to_calibration(servo_num):
        login_widget.close()
        window = FaceCalibrationWidget(servo_num=servo_num)
        window.show()
    login_ui.pushButton.clicked.connect(lambda: go_to_calibration(29))
    login_ui.pushButton_2.clicked.connect(lambda: go_to_calibration(27))
    login_ui.pushButton_3.clicked.connect(lambda: go_to_calibration(21))
    login_ui.pushButton_4.clicked.connect(lambda: go_to_calibration(17))

    sys.exit(app.exec())
