# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'login2.ui'
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
from PySide6.QtWidgets import (QApplication, QPushButton, QSizePolicy, QTextEdit,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(957, 498)
        Form.setStyleSheet(u"/* \u4e3b\u7a97\u53e3\uff1a\u4fee\u590dbackground-size\u62a5\u9519 + \u6d88\u9664\u9876\u90e8\u7a81\u5140 */\n"
"QWidget#Form {\n"
"	background-image: url(:/img/cover.png);\n"
"	background-repeat: no-repeat;	\n"
"	background-position: center center;\n"
"	background-attachment: fixed;\n"
"    font-family: \"Microsoft YaHei\", \"\u5fae\u8f6f\u96c5\u9ed1\", sans-serif;\n"
"    font-size: 15px;\n"
"    margin: 0px;\n"
"    padding: 0px;\n"
"}\n"
"\n"
"/* \u6587\u672c\u6846/\u4e0b\u62c9\u6846\uff1a\u8c03\u5927\u5b57\u4f53 + \u4fdd\u6301\u548c\u6309\u94ae\u7edf\u4e00\u98ce\u683c */\n"
"QLabel, QTextEdit, QComboBox {\n"
"   background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"        stop:0 rgba(0, 200, 230, 0.65),\n"
"        stop:0.25 rgb(255, 255, 255),\n"
"		stop:0.7 rgb(255, 255, 255),\n"
"        stop:1 rgba(0, 200, 230, 0.65));\n"
"    color: rgb(0, 0, 0);\n"
"    border: 2px solid rgba(0, 230, 255, 0.9);\n"
"    border-radius: 12px;\n"
"    padding: 10px 25px 15px ;\n"
"    min-width: 300"
                        "px;\n"
"    font-weight: bold;\n"
"    font-size: 20px; /* \u91cd\u70b9\uff1a\u4ece18px\u8c03\u523020px\uff0c\u5b57\u4f53\u66f4\u5927\u66f4\u9192\u76ee */\n"
"    outline: none;\n"
"    /* \u65b0\u589e\uff1a\u786e\u4fdd\u4e0b\u62c9\u6846\u6587\u5b57\u4e5f\u540c\u6b65\u53d8\u5927 */\n"
"    font-family: \"Microsoft YaHei\", \"\u5fae\u8f6f\u96c5\u9ed1\", sans-serif;\n"
"}\n"
"\n"
"/* \u6838\u5fc3\u6309\u94ae\uff1a\u4fdd\u7559\u539f\u6709\u6837\u5f0f */\n"
"QPushButton {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"        stop:0 rgba(0, 230, 255, 0.7),\n"
"        stop:0.5 rgba(0, 200, 230, 0.65),\n"
"        stop:1 rgba(0, 170, 200, 0.6));\n"
"    color: #FFFFFF;\n"
"    border: 2px solid rgba(0, 230, 255, 0.9);\n"
"    border-radius: 12px;\n"
"    padding: 20px 30px;\n"
"    min-width: 300px;\n"
"    font-weight: bold;\n"
"    font-size: 18px;\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"        stop:0 rgba(0, 255, "
                        "255, 0.8),\n"
"        stop:0.5 rgba(0, 220, 240, 0.75),\n"
"        stop:1 rgba(0, 190, 220, 0.7));\n"
"    border: 2px solid rgba(0, 255, 255, 1);\n"
"    color: #FFFFFF;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"        stop:0 rgba(0, 180, 210, 0.6),\n"
"        stop:0.5 rgba(0, 160, 190, 0.55),\n"
"        stop:1 rgba(0, 140, 170, 0.5));\n"
"    border: 2px solid rgba(0, 200, 230, 0.8);\n"
"    padding-top: 22px;\n"
"    padding-bottom: 18px;\n"
"}\n"
"\n"
"QPushButton:focus {\n"
"    border: 3px solid rgba(0, 255, 255, 1);\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background: rgba(200, 200, 200, 0.5);\n"
"    border-color: rgba(150, 150, 150, 0.6);\n"
"    color: rgba(100, 100, 100, 0.7);\n"
"}")
        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(300, 150, 364, 61))
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(300, 230, 364, 61))
        self.textEdit = QTextEdit(Form)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(280, 50, 401, 81))
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(300, 300, 364, 61))
        self.pushButton_4 = QPushButton(Form)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(300, 380, 364, 61))

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"3.0\u7248\u672c", None))
        self.pushButton_2.setText(QCoreApplication.translate("Form", u"Mini\u9ad8\u914d", None))
        self.textEdit.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei','\u5fae\u8f6f\u96c5\u9ed1','sans-serif'; font-size:20px; font-weight:700; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400;\">\u8bf7\u9009\u62e9\u60a8\u8981\u8c03\u8bd5\u7684\u4ea7\u54c1\u4fe1\u606f\uff1a</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Microsoft YaHei UI'; font-size"
                        ":9pt; font-weight:400;\">	\u642d\u8f7d\u81ea\u7814\u8235\u673a\u677f\u76843.0\u7248\u672c\u673a\u5668</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400;\">	\u642d\u8f7d\u81ea\u7814\u8235\u673a\u677f\u7684Mini\u7248\u672c\u673a\u5668<br /></span></p></body></html>", None))
        self.pushButton_3.setText(QCoreApplication.translate("Form", u"Mini\u4e2d\u914d", None))
        self.pushButton_4.setText(QCoreApplication.translate("Form", u"Mini\u4f4e\u914d", None))
    # retranslateUi

