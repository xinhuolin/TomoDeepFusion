# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AtomSeg_V1.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1003, 1077)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imagePath = QtWidgets.QTextEdit(self.centralwidget)
        self.imagePath.setEnabled(True)
        self.imagePath.setGeometry(QtCore.QRect(20, 30, 821, 31))
        self.imagePath.setObjectName("imagePath")
        self.open = QtWidgets.QPushButton(self.centralwidget)
        self.open.setGeometry(QtCore.QRect(880, 30, 91, 31))
        self.open.setObjectName("open")
        # self.ori = QtWidgets.QLabel(self.centralwidget)
        # self.ori.setGeometry(QtCore.QRect(20, 160, 410, 410))
        # self.ori.setFrameShape(QtWidgets.QFrame.Box)
        # self.ori.setText("")
        # self.ori.setScaledContents(True)
        # self.ori.setObjectName("ori")
        self.label_recon = QtWidgets.QLabel(self.centralwidget)
        self.label_recon.setGeometry(QtCore.QRect(20, 340, 410, 31))
        self.label_model = QtWidgets.QLabel(self.centralwidget)
        self.label_model.setGeometry(QtCore.QRect(440, 340, 410, 31))
        self.label_frame = QtWidgets.QLabel(self.centralwidget)
        self.label_frame.setGeometry(QtCore.QRect(20, 240, 100, 31))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(False)
        self.label_recon.setFont(font)
        self.label_recon.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_recon.setAlignment(QtCore.Qt.AlignCenter)
        self.label_recon.setObjectName("label_recon")
        self.label_model.setFont(font)
        self.label_model.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_model.setAlignment(QtCore.Qt.AlignCenter)
        self.label_model.setObjectName("label_model")
        self.label_frame.setFont(font)
        self.label_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.label_frame.setObjectName("label_model")
        self.recon_output = QtWidgets.QLabel(self.centralwidget)
        self.recon_output.setGeometry(QtCore.QRect(20, 380, 410, 410))
        self.recon_output.setFrameShape(QtWidgets.QFrame.Box)
        self.recon_output.setText("")
        self.recon_output.setScaledContents(True)
        self.recon_output.setObjectName("recon")
        self.model_output = QtWidgets.QLabel(self.centralwidget)
        self.model_output.setGeometry(QtCore.QRect(440, 380, 410, 410)) #(440, 160, 410, 410)
        self.model_output.setFrameShape(QtWidgets.QFrame.Box)
        self.model_output.setText("")
        self.model_output.setScaledContents(True)
        self.model_output.setObjectName("model_output")


        self.listData = QtWidgets.QComboBox(self.centralwidget)
        self.listData.setGeometry(QtCore.QRect(20, 70, 821, 31))
        self.listData.setObjectName("listData")


        self.modelPath = QtWidgets.QComboBox(self.centralwidget)
        self.modelPath.setGeometry(QtCore.QRect(20, 110, 821, 31))
        self.modelPath.setObjectName("modelPath")
        self.modelPath.addItem("")
        self.modelPath.addItem("")
        self.modelPath.addItem("")
        self.modelPath.addItem("")
        self.modelPath.addItem("")
        self.reconPath = QtWidgets.QComboBox(self.centralwidget)
        self.reconPath.setGeometry(QtCore.QRect(20, 150, 821, 31))
        self.reconPath.setObjectName("reconPath")
        self.reconPath.addItem("")
        self.reconPath.addItem("")
        self.reconPath.addItem("")
        self.use_cuda = QtWidgets.QCheckBox(self.centralwidget)
        self.use_cuda.setGeometry(QtCore.QRect(860, 110, 131, 21))
        self.use_cuda.setChecked(True)
        self.use_cuda.setObjectName("use_cuda")

        self.img_num_show = QtWidgets.QSpinBox(self.centralwidget)
        self.img_num_show.setGeometry(QtCore.QRect(20, 280, 60, 23))
        self.img_num = QtWidgets.QSlider(self.centralwidget)
        self.img_num.setGeometry(QtCore.QRect(20, 200, 821, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.img_num.setFont(font)
        self.img_num.setMinimum(0)  # 最小值
        # self.img_num.setMaximum(127)  # 最大值
        self.img_num.setSingleStep(1)  # 步长
        self.img_num.setOrientation(QtCore.Qt.Horizontal)
        self.img_num.setObjectName("img_num")


        self.load = QtWidgets.QPushButton(self.centralwidget)
        self.load.setGeometry(QtCore.QRect(880, 70, 91, 31))
        self.load.setObjectName("load")
        self.label_bin = QtWidgets.QLabel(self.centralwidget)
        self.label_bin.setGeometry(QtCore.QRect(870, 350, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(False)
        self.label_bin.setFont(font)
        self.label_bin.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_bin.setAlignment(QtCore.Qt.AlignCenter)
        self.label_bin.setObjectName("label")
        self.tobinary = QtWidgets.QPushButton(self.centralwidget)
        self.tobinary.setGeometry(QtCore.QRect(880, 440, 91, 31))
        self.tobinary.setObjectName("tobinary")





        self.set_iter_label= QtWidgets.QLabel( 'Iteration:' ,self.centralwidget)
        self.set_iter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.set_iter_label.setGeometry(QtCore.QRect(860, 210, 60, 23))

        self.set_iter = QtWidgets.QSpinBox(self.centralwidget)
        self.set_iter.setGeometry(QtCore.QRect(930, 210, 60, 23))
        self.set_iter.setRange(1, 5)  # 1
        self.set_iter.setSingleStep(1)  # 2
        self.set_iter.setValue(1)





        self.imagePath.raise_()
        self.open.raise_()
        self.recon_output.raise_()
        self.model_output.raise_()
        self.listData.raise_()
        self.modelPath.raise_()
        self.reconPath.raise_()
        self.img_num.raise_()
        self.label_recon.raise_()
        self.label_model.raise_()
        self.label_frame.raise_()
        self.load.raise_()
        self.label_bin.raise_()
        self.tobinary.raise_()
        self.use_cuda.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1003, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Atom Segmentation"))
        self.open.setText(_translate("MainWindow", "OPEN"))
        self.modelPath.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.modelPath.setItemText(0, _translate("MainWindow", "Denoise(49)"))
        self.modelPath.setItemText(1, _translate("MainWindow", "Denoise(80)"))
        self.modelPath.setItemText(2, _translate("MainWindow", "Denoise(sharp)"))
        self.modelPath.setItemText(3, _translate("MainWindow", "Deepfusion(3d)"))
        self.modelPath.setItemText(4, _translate("MainWindow", "Deepfusion(3dGAN)"))
        self.reconPath.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.reconPath.setItemText(0, _translate("MainWindow", "wbp"))
        self.reconPath.setItemText(1, _translate("MainWindow", "sart"))
        self.reconPath.setItemText(2, _translate("MainWindow", "sirt"))

        self.use_cuda.setText(_translate("MainWindow", "Use CUDA"))

        self.load.setText(_translate("MainWindow", "Calculate"))
        self.label_recon.setText(_translate("MainWindow", "Reconstruction Result"))
        self.label_model.setText(_translate("MainWindow", "Denoise Result"))
        self.label_frame.setText(_translate("MainWindow", "Frame Number"))
        self.tobinary.setText(_translate("MainWindow", "ToBinaryFile"))



