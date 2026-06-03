# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interface_17.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QGroupBox,
    QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QSlider, QSpinBox,
    QTextBrowser, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.setEnabled(True)
        Form.resize(1340, 845)
        Form.setStyleSheet(u"/* ========== GroupBox\uff1a\u5de6\u4e0a\u2192\u53f3\u4e0b \u6d45\u7070\u2192\u767d\u2192\u6d45\u7070 \u6e10\u53d8\uff08\u52a0\u6df1\u7248\uff09 ========== */\n"
"QGroupBox {\n"
"    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"                                stop:0 #dee2e6,    \n"
"                                stop:0.5 #ffffff,  \n"
"                                stop:1 #dee2e6);   \n"
"    border: 1px solid #d0d7e3;\n"
"    border-radius: 6px;\n"
"    margin-top: 12px;\n"
"    padding-top: 10px;\n"
"}\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    left: 10px;\n"
"    padding: 0 8px;\n"
"    color: #2c3e50;\n"
"    background-color: transparent;\n"
"}\n"
"\n"
"/* ========== \u5168\u5c40 QSpinBox \u6837\u5f0f ========== */\n"
"QSpinBox {\n"
"    min-width: 31px;\n"
"    max-width: 31px;\n"
"    height: 28px;\n"
"    border: 2px solid rgb(208, 208, 208);\n"
"    border-radius: 6px;\n"
"    padding: 2px 4px;\n"
"    background-color: #ffffff;\n"
"}\n"
"QSpinBox::up-button, QSpi"
                        "nBox::down-button {\n"
"    width: 20px;\n"
"}\n"
"QSpinBox[objectName*=\"M\"] {\n"
"    min-width: 51px;\n"
"    max-width: 51px;\n"
"	border: 2px solid rgb(170, 170, 170);\n"
"	border-radius: 4px;\n"
"}\n"
"\n"
"/* ========== \u6309\u94ae\u6837\u5f0f\uff1a\u65b0\u589e\u7acb\u4f53\u6e10\u53d8+\u6587\u5b57\u52a0\u7c97\uff0c\u5176\u4ed6\u4e0d\u53d8 ========== */\n"
"/* \u57fa\u7840\u6309\u94ae\uff08\u6d45\u84dd\u52a0\u6df1\u7248+\u7acb\u4f53\u6e10\u53d8+\u6587\u5b57\u52a0\u7c97\uff09 */\n"
"QPushButton {\n"
"    /* \u7acb\u4f53\u6e10\u53d8\u80cc\u666f\uff1a\u4e0a\u6d45\u4e0b\u6df1\uff0c\u589e\u5f3a\u51f8\u8d77\u611f */\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #e6f2ff, stop:0.5 #dceeff, stop:1 #d0e5ff);\n"
"    color: #1e88e5; /* \u6587\u5b57\u84dd\u52a0\u6df1 */\n"
"    font-weight: bold; /* \u6838\u5fc3\uff1a\u6587\u5b57\u52a0\u7c97 */\n"
"    border: 1px solid;\n"
"    border-color: #b9d9fb #9fc8f7 #9fc8f7 #b9d9fb; /* \u7acb\u4f53\u8fb9\u6846 */\n"
""
                        "    border-radius: 6px;\n"
"    padding: 6px 16px; /* \u4e0a\u4e0b\u5185\u8fb9\u8ddd\u51cf\u5c11\uff0c\u964d\u4f4e\u9ad8\u5ea6 */\n"
"    min-height: 28px; /* \u6309\u94ae\u9ad8\u5ea6\u4ece32\u219228px */\n"
"    min-width: 80px;\n"
"    font-size: 9pt;\n"
"}\n"
"\n"
"/* \u60ac\u505c\u72b6\u6001\uff1a\u6e10\u53d8\u52a0\u6df1+\u6587\u5b57\u52a0\u7c97\u4fdd\u7559 */\n"
"QPushButton:hover {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #d7e9ff, stop:0.5 #cbe4ff, stop:1 #bddcff);\n"
"    border-color: #9fc8f7 #80b7f5 #80b7f5 #9fc8f7;\n"
"}\n"
"\n"
"/* \u6309\u4e0b\u72b6\u6001\uff1a\u53cd\u5411\u6e10\u53d8+\u6587\u5b57\u52a0\u7c97\u4fdd\u7559 */\n"
"QPushButton:pressed {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #b8d8ff, stop:0.5 #c1ddff, stop:1 #cbe4ff);\n"
"    border-color: #80b7f5 #9fc8f7 #9fc8f7 #80b7f5;\n"
"}\n"
"\n"
"/* \u7981\u7528\u72b6\u6001\uff1a\u65e0\u6e10\u53d8+\u6587\u5b57\u52a0\u7c97"
                        "\u4fdd\u7559 */\n"
"QPushButton:disabled {\n"
"    background: #f8f9fa;\n"
"    color: #ced4da;\n"
"    font-weight: bold;\n"
"    border-color: #e9ecef;\n"
"}\n"
"\n"
"/* ========== \u9ad8\u4eae\u6309\u94ae\uff08\u4fdd\u5b58\u7c7b\uff1a\u6d45\u7eff\u6e10\u53d8+\u6587\u5b57\u52a0\u7c97\uff09 ========== */\n"
"QPushButton[objectName*=\"save\"], \n"
"QPushButton[objectName*=\"openORclose\"], \n"
"QPushButton[objectName*=\"confirm\"],\n"
"QPushButton[objectName*=\"ok\"] {\n"
"    /* \u6d45\u7eff\u7acb\u4f53\u6e10\u53d8 */\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #f0fae8, stop:0.5 #e6f7e0, stop:1 #dcf4d2);\n"
"    color: #389e0d; /* \u6587\u5b57\u7eff\u52a0\u6df1 */\n"
"    font-weight: bold; /* \u6587\u5b57\u52a0\u7c97 */\n"
"    border-color: #c5e8b7 #a3d98b #a3d98b #c5e8b7;\n"
"}\n"
"QPushButton[objectName*=\"save\"]:hover {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #e7f5d9, stop:0.5 #d7f"
                        "0cb, stop:1 #c8e7bc);\n"
"    border-color: #a3d98b #82c91e #82c91e #a3d98b;\n"
"}\n"
"QPushButton[objectName*=\"save\"]:pressed {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #c8e7bc, stop:0.5 #d0e9c4, stop:1 #d7f0cb);\n"
"    border-color: #82c91e #a3d98b #a3d98b #82c91e;\n"
"}\n"
"\n"
"/* ========== \u5371\u9669\u6309\u94ae\uff08\u91cd\u7f6e/\u590d\u539f\u7c7b\uff1a\u6d45\u7ea2\u6e10\u53d8+\u6587\u5b57\u52a0\u7c97\uff09 ========== */\n"
"QPushButton[objectName*=\"reset\"], \n"
"QPushButton[objectName*=\"delete\"],\n"
"QPushButton[objectName*=\"clear\"],\n"
"QPushButton[objectName*=\"\u590d\u539f\"] {\n"
"    /* \u6d45\u7ea2\u7acb\u4f53\u6e10\u53d8 */\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #fff0ee, stop:0.5 #ffe8e6, stop:1 #ffdedb);\n"
"    color: #cf1322; /* \u6587\u5b57\u7ea2\u52a0\u6df1 */\n"
"    font-weight: bold; /* \u6587\u5b57\u52a0\u7c97 */\n"
"    border-color: #ffccc7 #ff9a9e"
                        " #ff9a9e #ffccc7;\n"
"}\n"
"QPushButton[objectName*=\"reset\"]:hover {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #ffe9e6, stop:0.5 #ffd9d6, stop:1 #ffc9c6);\n"
"    border-color: #ff9a9e #ff4d4f #ff4d4f #ff9a9e;\n"
"}\n"
"QPushButton[objectName*=\"reset\"]:pressed {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, \n"
"                                stop:0 #ffc9c6, stop:0.5 #ffd0cd, stop:1 #ffd9d6);\n"
"    border-color: #ff4d4f #ff9a9e #ff9a9e #ff4d4f;\n"
"}")
        self.label_38 = QLabel(Form)
        self.label_38.setObjectName(u"label_38")
        self.label_38.setEnabled(False)
        self.label_38.setGeometry(QRect(610, 880, 1, 1))
        self.label_39 = QLabel(Form)
        self.label_39.setObjectName(u"label_39")
        self.label_39.setEnabled(False)
        self.label_39.setGeometry(QRect(620, 880, 1, 1))
        self.label_40 = QLabel(Form)
        self.label_40.setObjectName(u"label_40")
        self.label_40.setEnabled(False)
        self.label_40.setGeometry(QRect(620, 880, 1, 1))
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(1160, 0, 171, 841))
        self.label_41 = QLabel(self.groupBox)
        self.label_41.setObjectName(u"label_41")
        self.label_41.setGeometry(QRect(10, 200, 41, 20))
        self.label_serial_port_2 = QLabel(self.groupBox)
        self.label_serial_port_2.setObjectName(u"label_serial_port_2")
        self.label_serial_port_2.setGeometry(QRect(20, 20, 131, 31))
        font = QFont()
        font.setFamilies([u"Microsoft Himalaya"])
        self.label_serial_port_2.setFont(font)
        self.label_serial_port_2.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse|Qt.TextInteractionFlag.TextSelectableByMouse)
        self.pushButton_refresh_serial = QPushButton(self.groupBox)
        self.pushButton_refresh_serial.setObjectName(u"pushButton_refresh_serial")
        self.pushButton_refresh_serial.setGeometry(QRect(10, 90, 151, 42))
        self.comboBox_serial_port = QComboBox(self.groupBox)
        self.comboBox_serial_port.setObjectName(u"comboBox_serial_port")
        self.comboBox_serial_port.setGeometry(QRect(10, 50, 151, 31))
        self.pushButton_save = QPushButton(self.groupBox)
        self.pushButton_save.setObjectName(u"pushButton_save")
        self.pushButton_save.setGeometry(QRect(10, 280, 151, 42))
        self.lineEdit_config = QLineEdit(self.groupBox)
        self.lineEdit_config.setObjectName(u"lineEdit_config")
        self.lineEdit_config.setGeometry(QRect(50, 190, 111, 31))
        self.listWidget_file = QListWidget(self.groupBox)
        self.listWidget_file.setObjectName(u"listWidget_file")
        self.listWidget_file.setGeometry(QRect(10, 420, 151, 261))
        self.pushButton_refresh = QPushButton(self.groupBox)
        self.pushButton_refresh.setObjectName(u"pushButton_refresh")
        self.pushButton_refresh.setGeometry(QRect(10, 370, 151, 42))
        self.pushButton_load_state = QPushButton(self.groupBox)
        self.pushButton_load_state.setObjectName(u"pushButton_load_state")
        self.pushButton_load_state.setGeometry(QRect(10, 690, 151, 42))
        self.pushButton_reset_state = QPushButton(self.groupBox)
        self.pushButton_reset_state.setObjectName(u"pushButton_reset_state")
        self.pushButton_reset_state.setGeometry(QRect(10, 740, 151, 42))
        self.pushButton_save_state = QPushButton(self.groupBox)
        self.pushButton_save_state.setObjectName(u"pushButton_save_state")
        self.pushButton_save_state.setGeometry(QRect(10, 790, 151, 42))
        self.pushButton_load = QPushButton(self.groupBox)
        self.pushButton_load.setObjectName(u"pushButton_load")
        self.pushButton_load.setGeometry(QRect(10, 230, 151, 42))
        self.pushButton_openORclose_serial = QPushButton(self.groupBox)
        self.pushButton_openORclose_serial.setObjectName(u"pushButton_openORclose_serial")
        self.pushButton_openORclose_serial.setGeometry(QRect(10, 140, 151, 42))
        self.label_47 = QLabel(self.groupBox)
        self.label_47.setObjectName(u"label_47")
        self.label_47.setGeometry(QRect(10, 340, 41, 20))
        self.lineEdit_bs = QLineEdit(self.groupBox)
        self.lineEdit_bs.setObjectName(u"lineEdit_bs")
        self.lineEdit_bs.setGeometry(QRect(50, 330, 111, 31))
        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(0, 0, 1151, 841))
        self.Slider_10 = QSlider(self.groupBox_2)
        self.Slider_10.setObjectName(u"Slider_10")
        self.Slider_10.setGeometry(QRect(700, 250, 25, 100))
        self.Slider_10.setMaximum(270)
        self.Slider_10.setOrientation(Qt.Orientation.Vertical)
        self.Slider_10.setInvertedAppearance(True)
        self.textBrowser = QTextBrowser(self.groupBox_2)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(480, 20, 151, 51))
        font1 = QFont()
        font1.setPointSize(10)
        self.textBrowser.setFont(font1)
        self.spinBox_0_E = QSpinBox(self.groupBox_2)
        self.spinBox_0_E.setObjectName(u"spinBox_0_E")
        self.spinBox_0_E.setGeometry(QRect(670, 40, 43, 22))
        self.spinBox_0_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_0_E.setKeyboardTracking(True)
        self.spinBox_0_E.setMaximum(270)
        self.spinBox_0_E.setValue(270)
        self.label_30 = QLabel(self.groupBox_2)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setGeometry(QRect(850, 620, 21, 16))
        self.spinBox_0_M = QSpinBox(self.groupBox_2)
        self.spinBox_0_M.setObjectName(u"spinBox_0_M")
        self.spinBox_0_M.setGeometry(QRect(710, 40, 63, 22))
        self.spinBox_0_M.setMaximum(270)
        self.label_22 = QLabel(self.groupBox_2)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(590, 700, 21, 16))
        self.spinBox_25_E = QSpinBox(self.groupBox_2)
        self.spinBox_25_E.setObjectName(u"spinBox_25_E")
        self.spinBox_25_E.setGeometry(QRect(160, 660, 43, 22))
        self.spinBox_25_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_25_E.setKeyboardTracking(True)
        self.spinBox_25_E.setMaximum(270)
        self.spinBox_18_E = QSpinBox(self.groupBox_2)
        self.spinBox_18_E.setObjectName(u"spinBox_18_E")
        self.spinBox_18_E.setGeometry(QRect(450, 670, 43, 22))
        self.spinBox_18_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_18_E.setKeyboardTracking(True)
        self.spinBox_18_E.setMaximum(270)
        self.label_28 = QLabel(self.groupBox_2)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setGeometry(QRect(240, 570, 101, 16))
        self.spinBox_1_M = QSpinBox(self.groupBox_2)
        self.spinBox_1_M.setObjectName(u"spinBox_1_M")
        self.spinBox_1_M.setGeometry(QRect(380, 40, 63, 22))
        self.spinBox_1_M.setMaximum(270)
        self.label_12 = QLabel(self.groupBox_2)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(670, 210, 111, 16))
        self.spinBox_8_M = QSpinBox(self.groupBox_2)
        self.spinBox_8_M.setObjectName(u"spinBox_8_M")
        self.spinBox_8_M.setGeometry(QRect(990, 280, 63, 22))
        self.spinBox_8_M.setMaximum(270)
        self.spinBox_1_E = QSpinBox(self.groupBox_2)
        self.spinBox_1_E.setObjectName(u"spinBox_1_E")
        self.spinBox_1_E.setGeometry(QRect(340, 40, 43, 22))
        self.spinBox_1_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_1_E.setKeyboardTracking(True)
        self.spinBox_1_E.setMaximum(270)
        self.spinBox_9_E = QSpinBox(self.groupBox_2)
        self.spinBox_9_E.setObjectName(u"spinBox_9_E")
        self.spinBox_9_E.setGeometry(QRect(50, 230, 43, 22))
        self.spinBox_9_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_9_E.setKeyboardTracking(True)
        self.spinBox_9_E.setMaximum(270)
        self.spinBox_22_E = QSpinBox(self.groupBox_2)
        self.spinBox_22_E.setObjectName(u"spinBox_22_E")
        self.spinBox_22_E.setGeometry(QRect(90, 590, 43, 22))
        self.spinBox_22_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_22_E.setKeyboardTracking(True)
        self.spinBox_22_E.setMaximum(270)
        self.spinBox_22_E.setValue(270)
        self.label_14 = QLabel(self.groupBox_2)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(760, 270, 111, 16))
        self.spinBox_10_M = QSpinBox(self.groupBox_2)
        self.spinBox_10_M.setObjectName(u"spinBox_10_M")
        self.spinBox_10_M.setGeometry(QRect(710, 230, 63, 22))
        self.spinBox_10_M.setMaximum(270)
        self.spinBox_22_S = QSpinBox(self.groupBox_2)
        self.spinBox_22_S.setObjectName(u"spinBox_22_S")
        self.spinBox_22_S.setGeometry(QRect(240, 590, 43, 22))
        self.spinBox_22_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_22_S.setKeyboardTracking(True)
        self.spinBox_22_S.setMaximum(270)
        self.label_34 = QLabel(self.groupBox_2)
        self.label_34.setObjectName(u"label_34")
        self.label_34.setGeometry(QRect(230, 620, 21, 16))
        self.spinBox_8_E = QSpinBox(self.groupBox_2)
        self.spinBox_8_E.setObjectName(u"spinBox_8_E")
        self.spinBox_8_E.setGeometry(QRect(1040, 220, 43, 22))
        self.spinBox_8_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_8_E.setKeyboardTracking(True)
        self.spinBox_8_E.setMaximum(270)
        self.spinBox_8_E.setValue(270)
        self.Slider_9 = QSlider(self.groupBox_2)
        self.Slider_9.setObjectName(u"Slider_9")
        self.Slider_9.setGeometry(QRect(60, 250, 25, 100))
        self.Slider_9.setMaximum(270)
        self.Slider_9.setOrientation(Qt.Orientation.Vertical)
        self.Slider_9.setInvertedAppearance(True)
        self.Slider_19 = QSlider(self.groupBox_2)
        self.Slider_19.setObjectName(u"Slider_19")
        self.Slider_19.setGeometry(QRect(530, 630, 22, 100))
        self.Slider_19.setMaximum(270)
        self.Slider_19.setOrientation(Qt.Orientation.Vertical)
        self.Slider_19.setInvertedAppearance(True)
        self.Slider_8 = QSlider(self.groupBox_2)
        self.Slider_8.setObjectName(u"Slider_8")
        self.Slider_8.setGeometry(QRect(1050, 249, 22, 91))
        self.Slider_8.setMaximum(270)
        self.Slider_8.setOrientation(Qt.Orientation.Vertical)
        self.Slider_1 = QSlider(self.groupBox_2)
        self.Slider_1.setObjectName(u"Slider_1")
        self.Slider_1.setGeometry(QRect(370, 60, 22, 100))
        self.Slider_1.setMaximum(270)
        self.Slider_1.setOrientation(Qt.Orientation.Vertical)
        self.Slider_1.setInvertedAppearance(True)
        self.label_37 = QLabel(self.groupBox_2)
        self.label_37.setObjectName(u"label_37")
        self.label_37.setEnabled(False)
        self.label_37.setGeometry(QRect(120, 770, 1, 1))
        self.label_18 = QLabel(self.groupBox_2)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(590, 500, 21, 16))
        self.spinBox_26_E = QSpinBox(self.groupBox_2)
        self.spinBox_26_E.setObjectName(u"spinBox_26_E")
        self.spinBox_26_E.setGeometry(QRect(750, 790, 43, 22))
        self.spinBox_26_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_26_E.setKeyboardTracking(True)
        self.spinBox_26_E.setMaximum(270)
        self.spinBox_24_E = QSpinBox(self.groupBox_2)
        self.spinBox_24_E.setObjectName(u"spinBox_24_E")
        self.spinBox_24_E.setGeometry(QRect(800, 590, 43, 22))
        self.spinBox_24_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_24_E.setKeyboardTracking(True)
        self.spinBox_24_E.setMaximum(270)
        self.spinBox_24_E.setValue(270)
        self.spinBox_22_M = QSpinBox(self.groupBox_2)
        self.spinBox_22_M.setObjectName(u"spinBox_22_M")
        self.spinBox_22_M.setGeometry(QRect(280, 590, 63, 22))
        self.spinBox_22_M.setMaximum(270)
        self.spinBox_16_M = QSpinBox(self.groupBox_2)
        self.spinBox_16_M.setObjectName(u"spinBox_16_M")
        self.spinBox_16_M.setGeometry(QRect(540, 410, 63, 22))
        self.spinBox_16_M.setMaximum(270)
        self.spinBox_13_S = QSpinBox(self.groupBox_2)
        self.spinBox_13_S.setObjectName(u"spinBox_13_S")
        self.spinBox_13_S.setGeometry(QRect(280, 290, 43, 22))
        self.spinBox_13_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_13_S.setKeyboardTracking(True)
        self.spinBox_13_S.setMaximum(270)
        self.spinBox_13_S.setValue(270)
        self.spinBox_23_M = QSpinBox(self.groupBox_2)
        self.spinBox_23_M.setObjectName(u"spinBox_23_M")
        self.spinBox_23_M.setGeometry(QRect(890, 530, 63, 22))
        self.spinBox_23_M.setMaximum(270)
        self.label_23 = QLabel(self.groupBox_2)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(470, 700, 21, 16))
        self.spinBox_0_S = QSpinBox(self.groupBox_2)
        self.spinBox_0_S.setObjectName(u"spinBox_0_S")
        self.spinBox_0_S.setGeometry(QRect(690, 160, 43, 22))
        self.spinBox_0_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_0_S.setKeyboardTracking(True)
        self.spinBox_0_S.setMaximum(270)
        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(670, 20, 111, 20))
        self.label_11 = QLabel(self.groupBox_2)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(40, 210, 101, 16))
        self.label_33 = QLabel(self.groupBox_2)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setGeometry(QRect(140, 510, 121, 16))
        self.spinBox_8_S = QSpinBox(self.groupBox_2)
        self.spinBox_8_S.setObjectName(u"spinBox_8_S")
        self.spinBox_8_S.setGeometry(QRect(1040, 340, 43, 22))
        self.spinBox_8_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_8_S.setKeyboardTracking(True)
        self.spinBox_8_S.setMaximum(270)
        self.spinBox_23_E = QSpinBox(self.groupBox_2)
        self.spinBox_23_E.setObjectName(u"spinBox_23_E")
        self.spinBox_23_E.setGeometry(QRect(850, 530, 43, 22))
        self.spinBox_23_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_23_E.setKeyboardTracking(True)
        self.spinBox_23_E.setMaximum(270)
        self.spinBox_23_E.setValue(270)
        self.spinBox_17_E = QSpinBox(self.groupBox_2)
        self.spinBox_17_E.setObjectName(u"spinBox_17_E")
        self.spinBox_17_E.setGeometry(QRect(450, 470, 43, 22))
        self.spinBox_17_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_17_E.setKeyboardTracking(True)
        self.spinBox_17_E.setMaximum(270)
        self.spinBox_17_E.setValue(270)
        self.label_36 = QLabel(self.groupBox_2)
        self.label_36.setObjectName(u"label_36")
        self.label_36.setGeometry(QRect(770, 700, 121, 16))
        self.spinBox_12_E = QSpinBox(self.groupBox_2)
        self.spinBox_12_E.setObjectName(u"spinBox_12_E")
        self.spinBox_12_E.setGeometry(QRect(760, 290, 43, 22))
        self.spinBox_12_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_12_E.setKeyboardTracking(True)
        self.spinBox_12_E.setMaximum(270)
        self.spinBox_12_E.setValue(270)
        self.spinBox_11_E = QSpinBox(self.groupBox_2)
        self.spinBox_11_E.setObjectName(u"spinBox_11_E")
        self.spinBox_11_E.setGeometry(QRect(340, 230, 43, 22))
        self.spinBox_11_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_11_E.setKeyboardTracking(True)
        self.spinBox_11_E.setMaximum(270)
        self.Slider_24 = QSlider(self.groupBox_2)
        self.Slider_24.setObjectName(u"Slider_24")
        self.Slider_24.setGeometry(QRect(840, 590, 100, 25))
        self.Slider_24.setMaximum(270)
        self.Slider_24.setOrientation(Qt.Orientation.Horizontal)
        self.Slider_16 = QSlider(self.groupBox_2)
        self.Slider_16.setObjectName(u"Slider_16")
        self.Slider_16.setGeometry(QRect(530, 430, 22, 100))
        self.Slider_16.setMaximum(270)
        self.Slider_16.setOrientation(Qt.Orientation.Vertical)
        self.Slider_16.setInvertedAppearance(True)
        self.spinBox_9_S = QSpinBox(self.groupBox_2)
        self.spinBox_9_S.setObjectName(u"spinBox_9_S")
        self.spinBox_9_S.setGeometry(QRect(50, 350, 43, 22))
        self.spinBox_9_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_9_S.setKeyboardTracking(True)
        self.spinBox_9_S.setMaximum(270)
        self.spinBox_9_S.setValue(270)
        self.spinBox_12_M = QSpinBox(self.groupBox_2)
        self.spinBox_12_M.setObjectName(u"spinBox_12_M")
        self.spinBox_12_M.setGeometry(QRect(800, 290, 63, 22))
        self.spinBox_12_M.setMaximum(270)
        self.spinBox_16_E = QSpinBox(self.groupBox_2)
        self.spinBox_16_E.setObjectName(u"spinBox_16_E")
        self.spinBox_16_E.setGeometry(QRect(520, 530, 43, 22))
        self.spinBox_16_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_16_E.setKeyboardTracking(True)
        self.spinBox_16_E.setMaximum(270)
        self.spinBox_16_E.setValue(270)
        self.spinBox_18_S = QSpinBox(self.groupBox_2)
        self.spinBox_18_S.setObjectName(u"spinBox_18_S")
        self.spinBox_18_S.setGeometry(QRect(590, 670, 43, 22))
        self.spinBox_18_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_18_S.setKeyboardTracking(True)
        self.spinBox_18_S.setMaximum(270)
        self.spinBox_18_S.setValue(270)
        self.label_15 = QLabel(self.groupBox_2)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(440, 270, 111, 16))
        self.label_25 = QLabel(self.groupBox_2)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(500, 590, 121, 16))
        self.spinBox_1_S = QSpinBox(self.groupBox_2)
        self.spinBox_1_S.setObjectName(u"spinBox_1_S")
        self.spinBox_1_S.setGeometry(QRect(360, 160, 43, 22))
        self.spinBox_1_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_1_S.setKeyboardTracking(True)
        self.spinBox_1_S.setMaximum(270)
        self.spinBox_1_S.setValue(270)
        self.spinBox_17_M = QSpinBox(self.groupBox_2)
        self.spinBox_17_M.setObjectName(u"spinBox_17_M")
        self.spinBox_17_M.setGeometry(QRect(640, 470, 63, 22))
        self.spinBox_17_M.setMaximum(270)
        self.spinBox_24_M = QSpinBox(self.groupBox_2)
        self.spinBox_24_M.setObjectName(u"spinBox_24_M")
        self.spinBox_24_M.setGeometry(QRect(980, 590, 63, 22))
        self.spinBox_24_M.setMaximum(270)
        self.spinBox_25_M = QSpinBox(self.groupBox_2)
        self.spinBox_25_M.setObjectName(u"spinBox_25_M")
        self.spinBox_25_M.setGeometry(QRect(180, 530, 63, 22))
        self.spinBox_25_M.setMaximum(270)
        self.spinBox_11_M = QSpinBox(self.groupBox_2)
        self.spinBox_11_M.setObjectName(u"spinBox_11_M")
        self.spinBox_11_M.setGeometry(QRect(380, 230, 63, 22))
        self.spinBox_11_M.setMaximum(270)
        self.spinBox_10_E = QSpinBox(self.groupBox_2)
        self.spinBox_10_E.setObjectName(u"spinBox_10_E")
        self.spinBox_10_E.setGeometry(QRect(690, 350, 43, 22))
        self.spinBox_10_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_10_E.setKeyboardTracking(True)
        self.spinBox_10_E.setMaximum(270)
        self.spinBox_10_E.setValue(270)
        self.Slider_22 = QSlider(self.groupBox_2)
        self.Slider_22.setObjectName(u"Slider_22")
        self.Slider_22.setGeometry(QRect(140, 590, 100, 25))
        self.Slider_22.setMaximum(270)
        self.Slider_22.setOrientation(Qt.Orientation.Horizontal)
        self.Slider_22.setInvertedAppearance(True)
        self.label_20 = QLabel(self.groupBox_2)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(500, 390, 121, 16))
        self.Slider_18 = QSlider(self.groupBox_2)
        self.Slider_18.setObjectName(u"Slider_18")
        self.Slider_18.setGeometry(QRect(490, 670, 100, 25))
        self.Slider_18.setMaximum(270)
        self.Slider_18.setOrientation(Qt.Orientation.Horizontal)
        self.Slider_27 = QSlider(self.groupBox_2)
        self.Slider_27.setObjectName(u"Slider_27")
        self.Slider_27.setEnabled(False)
        self.Slider_27.setGeometry(QRect(120, 770, 1, 1))
        self.Slider_27.setMaximum(270)
        self.Slider_27.setOrientation(Qt.Orientation.Vertical)
        self.Slider_27.setInvertedAppearance(True)
        self.label_29 = QLabel(self.groupBox_2)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setGeometry(QRect(850, 510, 121, 16))
        self.spinBox_19_M = QSpinBox(self.groupBox_2)
        self.spinBox_19_M.setObjectName(u"spinBox_19_M")
        self.spinBox_19_M.setGeometry(QRect(540, 610, 63, 22))
        self.spinBox_19_M.setMaximum(270)
        self.spinBox_19_E = QSpinBox(self.groupBox_2)
        self.spinBox_19_E.setObjectName(u"spinBox_19_E")
        self.spinBox_19_E.setGeometry(QRect(520, 730, 43, 22))
        self.spinBox_19_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_19_E.setKeyboardTracking(True)
        self.spinBox_19_E.setMaximum(270)
        self.spinBox_19_E.setValue(270)
        self.spinBox_9_M = QSpinBox(self.groupBox_2)
        self.spinBox_9_M.setObjectName(u"spinBox_9_M")
        self.spinBox_9_M.setGeometry(QRect(80, 290, 63, 22))
        self.spinBox_9_M.setMaximum(270)
        self.spinBox_24_S = QSpinBox(self.groupBox_2)
        self.spinBox_24_S.setObjectName(u"spinBox_24_S")
        self.spinBox_24_S.setGeometry(QRect(940, 590, 43, 22))
        self.spinBox_24_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_24_S.setKeyboardTracking(True)
        self.spinBox_24_S.setMaximum(270)
        self.Slider_11 = QSlider(self.groupBox_2)
        self.Slider_11.setObjectName(u"Slider_11")
        self.Slider_11.setGeometry(QRect(370, 250, 25, 100))
        self.Slider_11.setMaximum(270)
        self.Slider_11.setOrientation(Qt.Orientation.Vertical)
        self.Slider_11.setInvertedAppearance(True)
        self.spinBox_26_S = QSpinBox(self.groupBox_2)
        self.spinBox_26_S.setObjectName(u"spinBox_26_S")
        self.spinBox_26_S.setGeometry(QRect(750, 720, 43, 22))
        self.spinBox_26_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_26_S.setKeyboardTracking(True)
        self.spinBox_26_S.setMaximum(270)
        self.spinBox_26_S.setValue(270)
        self.Slider_0 = QSlider(self.groupBox_2)
        self.Slider_0.setObjectName(u"Slider_0")
        self.Slider_0.setGeometry(QRect(700, 69, 22, 91))
        self.Slider_0.setMaximum(270)
        self.Slider_0.setOrientation(Qt.Orientation.Vertical)
        self.Slider_12 = QSlider(self.groupBox_2)
        self.Slider_12.setObjectName(u"Slider_12")
        self.Slider_12.setGeometry(QRect(650, 290, 111, 25))
        self.Slider_12.setMaximum(270)
        self.Slider_12.setOrientation(Qt.Orientation.Horizontal)
        self.spinBox_10_S = QSpinBox(self.groupBox_2)
        self.spinBox_10_S.setObjectName(u"spinBox_10_S")
        self.spinBox_10_S.setGeometry(QRect(670, 230, 43, 22))
        self.spinBox_10_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_10_S.setKeyboardTracking(True)
        self.spinBox_10_S.setMaximum(270)
        self.Slider_23 = QSlider(self.groupBox_2)
        self.Slider_23.setObjectName(u"Slider_23")
        self.Slider_23.setGeometry(QRect(880, 550, 22, 100))
        self.Slider_23.setMaximum(270)
        self.Slider_23.setOrientation(Qt.Orientation.Vertical)
        self.Slider_23.setInvertedAppearance(True)
        self.label_24 = QLabel(self.groupBox_2)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setGeometry(QRect(590, 650, 121, 16))
        self.spinBox_25_S = QSpinBox(self.groupBox_2)
        self.spinBox_25_S.setObjectName(u"spinBox_25_S")
        self.spinBox_25_S.setGeometry(QRect(140, 530, 43, 22))
        self.spinBox_25_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_25_S.setKeyboardTracking(True)
        self.spinBox_25_S.setMaximum(270)
        self.spinBox_25_S.setValue(270)
        self.spinBox_13_M = QSpinBox(self.groupBox_2)
        self.spinBox_13_M.setObjectName(u"spinBox_13_M")
        self.spinBox_13_M.setGeometry(QRect(470, 290, 63, 22))
        self.spinBox_13_M.setMaximum(270)
        self.spinBox_11_S = QSpinBox(self.groupBox_2)
        self.spinBox_11_S.setObjectName(u"spinBox_11_S")
        self.spinBox_11_S.setGeometry(QRect(360, 350, 43, 22))
        self.spinBox_11_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_11_S.setKeyboardTracking(True)
        self.spinBox_11_S.setMaximum(270)
        self.spinBox_11_S.setValue(270)
        self.label_21 = QLabel(self.groupBox_2)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(600, 450, 121, 16))
        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(350, 20, 101, 20))
        self.label_19 = QLabel(self.groupBox_2)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(490, 500, 21, 16))
        self.Slider_26 = QSlider(self.groupBox_2)
        self.Slider_26.setObjectName(u"Slider_26")
        self.Slider_26.setGeometry(QRect(800, 720, 22, 100))
        self.Slider_26.setMaximum(270)
        self.Slider_26.setOrientation(Qt.Orientation.Vertical)
        self.Slider_13 = QSlider(self.groupBox_2)
        self.Slider_13.setObjectName(u"Slider_13")
        self.Slider_13.setGeometry(QRect(330, 290, 100, 25))
        self.Slider_13.setMaximum(270)
        self.Slider_13.setOrientation(Qt.Orientation.Horizontal)
        self.Slider_13.setInvertedAppearance(True)
        self.spinBox_19_S = QSpinBox(self.groupBox_2)
        self.spinBox_19_S.setObjectName(u"spinBox_19_S")
        self.spinBox_19_S.setGeometry(QRect(500, 610, 43, 22))
        self.spinBox_19_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_19_S.setKeyboardTracking(True)
        self.spinBox_19_S.setMaximum(270)
        self.spinBox_16_S = QSpinBox(self.groupBox_2)
        self.spinBox_16_S.setObjectName(u"spinBox_16_S")
        self.spinBox_16_S.setGeometry(QRect(500, 410, 43, 22))
        self.spinBox_16_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_16_S.setKeyboardTracking(True)
        self.spinBox_16_S.setMaximum(270)
        self.label_31 = QLabel(self.groupBox_2)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setGeometry(QRect(930, 620, 21, 16))
        self.spinBox_12_S = QSpinBox(self.groupBox_2)
        self.spinBox_12_S.setObjectName(u"spinBox_12_S")
        self.spinBox_12_S.setGeometry(QRect(610, 290, 43, 22))
        self.spinBox_12_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_12_S.setKeyboardTracking(True)
        self.spinBox_12_S.setMaximum(270)
        self.spinBox_13_E = QSpinBox(self.groupBox_2)
        self.spinBox_13_E.setObjectName(u"spinBox_13_E")
        self.spinBox_13_E.setGeometry(QRect(430, 290, 43, 22))
        self.spinBox_13_E.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_13_E.setKeyboardTracking(True)
        self.spinBox_13_E.setMaximum(270)
        self.label_13 = QLabel(self.groupBox_2)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(340, 210, 111, 16))
        self.label_10 = QLabel(self.groupBox_2)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(1020, 200, 101, 16))
        self.label_35 = QLabel(self.groupBox_2)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setGeometry(QRect(130, 620, 21, 16))
        self.label_32 = QLabel(self.groupBox_2)
        self.label_32.setObjectName(u"label_32")
        self.label_32.setGeometry(QRect(940, 570, 121, 16))
        self.Slider_28 = QSlider(self.groupBox_2)
        self.Slider_28.setObjectName(u"Slider_28")
        self.Slider_28.setEnabled(False)
        self.Slider_28.setGeometry(QRect(360, 820, 1, 1))
        self.Slider_28.setMaximum(270)
        self.Slider_28.setOrientation(Qt.Orientation.Horizontal)
        self.Slider_28.setInvertedAppearance(True)
        self.spinBox_17_S = QSpinBox(self.groupBox_2)
        self.spinBox_17_S.setObjectName(u"spinBox_17_S")
        self.spinBox_17_S.setGeometry(QRect(600, 470, 43, 22))
        self.spinBox_17_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_17_S.setKeyboardTracking(True)
        self.spinBox_17_S.setMaximum(270)
        self.spinBox_23_S = QSpinBox(self.groupBox_2)
        self.spinBox_23_S.setObjectName(u"spinBox_23_S")
        self.spinBox_23_S.setGeometry(QRect(870, 650, 43, 22))
        self.spinBox_23_S.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinBox_23_S.setKeyboardTracking(True)
        self.spinBox_23_S.setMaximum(270)
        self.spinBox_26_M = QSpinBox(self.groupBox_2)
        self.spinBox_26_M.setObjectName(u"spinBox_26_M")
        self.spinBox_26_M.setGeometry(QRect(820, 750, 63, 22))
        self.spinBox_26_M.setMaximum(270)
        self.spinBox_18_M = QSpinBox(self.groupBox_2)
        self.spinBox_18_M.setObjectName(u"spinBox_18_M")
        self.spinBox_18_M.setGeometry(QRect(630, 670, 63, 22))
        self.spinBox_18_M.setMaximum(270)
        self.Slider_25 = QSlider(self.groupBox_2)
        self.Slider_25.setObjectName(u"Slider_25")
        self.Slider_25.setGeometry(QRect(170, 560, 25, 100))
        self.Slider_25.setMaximum(270)
        self.Slider_25.setOrientation(Qt.Orientation.Vertical)
        self.Slider_17 = QSlider(self.groupBox_2)
        self.Slider_17.setObjectName(u"Slider_17")
        self.Slider_17.setGeometry(QRect(500, 470, 100, 25))
        self.Slider_17.setMaximum(270)
        self.Slider_17.setOrientation(Qt.Orientation.Horizontal)
        self.Slider_17.setInvertedAppearance(True)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Face Calibration", None))
        Form.setProperty(u"title", QCoreApplication.translate("Form", u"GroupBox", None))
        self.label_38.setText(QCoreApplication.translate("Form", u"A28:\u65e0", None))
        self.label_39.setText(QCoreApplication.translate("Form", u"\u5de6", None))
        self.label_40.setText(QCoreApplication.translate("Form", u"\u53f3", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"\u8bbe\u7f6e", None))
        self.label_41.setText(QCoreApplication.translate("Form", u"\u914d  \u7f6e:", None))
        self.label_serial_port_2.setText(QCoreApplication.translate("Form", u"\u4e32\u53e3\u53f7", None))
        self.pushButton_refresh_serial.setText(QCoreApplication.translate("Form", u"\u5237\u65b0\u4e32\u53e3", None))
        self.pushButton_save.setText(QCoreApplication.translate("Form", u"\u4fdd\u5b58\u914d\u7f6e", None))
        self.pushButton_refresh.setText(QCoreApplication.translate("Form", u"\u5237\u65b0bs\u6587\u4ef6", None))
        self.pushButton_load_state.setText(QCoreApplication.translate("Form", u"\u8f7d\u5165\u52a8\u4f5c", None))
        self.pushButton_reset_state.setText(QCoreApplication.translate("Form", u"\u590d\u539f\u521d\u59cb\u52a8\u4f5c", None))
        self.pushButton_save_state.setText(QCoreApplication.translate("Form", u"\u4fdd\u5b58\u5f53\u524d\u52a8\u4f5c", None))
        self.pushButton_load.setText(QCoreApplication.translate("Form", u"\u8f7d\u5165\u914d\u7f6e", None))
        self.pushButton_openORclose_serial.setText(QCoreApplication.translate("Form", u"\u6253\u5f00\u4e32\u53e3", None))
        self.label_47.setText(QCoreApplication.translate("Form", u"\u8868  \u60c5:", None))
        self.lineEdit_bs.setText("")
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"\u63a7\u5236", None))
        self.textBrowser.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'.AppleSystemUIFont'; font-size:13pt;\">\u673a\u5668\u4eba\u5934\u9762\u671d\u4f60\u65f6\uff0c\u62d6\u52a8\u7684\u65b9\u5411\u548c\u673a\u5668\u70b9\u4f4d\u8fd0\u52a8\u65b9\u5411\u4e00\u81f4</span></p></body></html>", None))
        self.label_30.setText(QCoreApplication.translate("Form", u"\u524d", None))
        self.label_22.setText(QCoreApplication.translate("Form", u"\u540e", None))
        self.label_28.setText(QCoreApplication.translate("Form", u"A22:\u53f3\u5634\u89d2\u524d\u2014\u540e", None))
        self.label_12.setText(QCoreApplication.translate("Form", u"A10:\u5de6\u773c\u7403 \u4e0b\u2014\u4e0a", None))
        self.label_14.setText(QCoreApplication.translate("Form", u"A12:\u5de6\u773c\u7403 \u5185\u2014\u5916", None))
        self.label_34.setText(QCoreApplication.translate("Form", u"\u524d", None))
        self.label_37.setText(QCoreApplication.translate("Form", u"A27:\u65e0", None))
        self.label_18.setText(QCoreApplication.translate("Form", u"\u540e", None))
        self.label_23.setText(QCoreApplication.translate("Form", u"\u524d", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"A0:\u5de6\u7709\u5934 \u4e0b\u2014\u4e0a", None))
        self.label_11.setText(QCoreApplication.translate("Form", u"A9:\u53f3\u773c\u76ae \u5f00\u2014\u5408", None))
        self.label_33.setText(QCoreApplication.translate("Form", u"A25:\u53f3\u5634\u89d2 \u4e0a\u2014\u4e0b", None))
        self.label_36.setText(QCoreApplication.translate("Form", u"A26:\u4e0b\u5df4 \u5f00\u2014\u5408", None))
        self.label_15.setText(QCoreApplication.translate("Form", u"A13:\u53f3\u773c\u7403 \u5185\u2014\u5916", None))
        self.label_25.setText(QCoreApplication.translate("Form", u"A19:\u4e2d\u4e0b\u5634\u5507 \u4e0a\u2014\u4e0b", None))
        self.label_20.setText(QCoreApplication.translate("Form", u"A16:\u4e2d\u4e0a\u5634\u5507 \u4e0a\u2014\u4e0b", None))
        self.label_29.setText(QCoreApplication.translate("Form", u"A23:\u5de6\u5634\u89d2 \u4e0a\u2014\u4e0b", None))
        self.label_24.setText(QCoreApplication.translate("Form", u"A18:\u4e2d\u4e0b\u5634\u5507 \u524d\u2014\u540e", None))
        self.label_21.setText(QCoreApplication.translate("Form", u"A17:\u4e2d\u4e0a\u5634\u5507 \u540e\u2014\u524d", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"A1:\u53f3\u7709\u5934 \u4e0b\u2014\u4e0a", None))
        self.label_19.setText(QCoreApplication.translate("Form", u"\u524d", None))
        self.label_31.setText(QCoreApplication.translate("Form", u"\u540e", None))
        self.label_13.setText(QCoreApplication.translate("Form", u"A11:\u53f3\u773c\u7403 \u4e0b\u2014\u4e0a", None))
        self.label_10.setText(QCoreApplication.translate("Form", u"A8:\u5de6\u773c\u76ae \u5f00\u2014\u5408", None))
        self.label_35.setText(QCoreApplication.translate("Form", u"\u540e", None))
        self.label_32.setText(QCoreApplication.translate("Form", u"A24:\u5de6\u5634\u89d2 \u524d\u2014\u540e", None))
    # retranslateUi

