#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
from os.path import exists

import numpy as np
import scipy.io as scio
from PIL import Image, ImageDraw
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import opening, watershed, disk, erosion

from UI_files.WinUI import Ui_MainWindow
from utils.utils import GetIndexRangeOfBlk, load_model, PIL2Pixmap, map01

import scipy.io as scio

class Code_MainWindow(Ui_MainWindow):
    def __init__(self, parent=None):
        super(Code_MainWindow, self).__init__()

        self.setupUi(self)
        self.open.clicked.connect(self.BrowseFolder)
        self.load.clicked.connect(self.LoadModel)
        self.img_num.valueChanged.connect(self.ResultShow)
        self.show_flag = False
        self.tobinary.clicked.connect(self.ArraytoBinary)


        self.__curdir = os.getcwd()  # current directory

        self.ori_image = None
        self.ori_content = None  # original image, PIL format
        self.output_image = None  # output image of model, PIL format
        self.ori_markers = None  # for saving usage, it's a rgb image of original, and with detection result on it
        self.out_markers = None  # for saving usage, it's a rgb image of result after denoising, and with detection result on it
        self.model_output_content = None  # 2d array of model output

        self.result = None
        self.denoised_image = None
        self.props = None

        self.imarray_original = None

        self.__model_dir = "model_weights"
        self.__models = {
            'Denoise(delta10)': os.path.join(self.__model_dir, 'Denoise(delta10).pth'),
            'Denoise(80)': os.path.join(self.__model_dir, 'Denoise(80).pth'),
            'Denoise(sharp)': os.path.join(self.__model_dir, 'Denoise(sharp).pth'),
            'Deepfusion(3d)': os.path.join(self.__model_dir, 'Deepfusion(3d).pth'),
            'Deepfusion(3dGAN)': os.path.join(self.__model_dir, 'Deepfusion(3dGAN).pth'),
            'Denoise(IC)': os.path.join(self.__model_dir, 'Denoise(IC).pth'),
        }

        from torch.cuda import is_available
        self.use_cuda.setChecked(is_available())
        self.use_cuda.setDisabled(not is_available())
        self.model_name = None
        self.recon_name = None
        self.imagePath_content = None

    def BrowseFolder(self):
        self.imagePath_content, _ = QFileDialog.getOpenFileName(self,
                                                                "open",
                                                                "/media/Elements/hewei/TomoFillNet/",
                                                                "All Files (*);; Image Files (*.png *.tif *.jpg *.ser *.dm3)")
        self.listData.clear()
        if self.imagePath_content:
            self.imagePath.setText(self.imagePath_content)
            file_name = os.path.basename(self.imagePath_content)
            _, suffix = os.path.splitext(file_name)
            if suffix == '.mat':
                self.matdata = scio.loadmat(self.imagePath_content)
                self.matkeys = []
                self.matangles = None
                self.matsize = None

                self.imarray_original = np.array(self.ori_image)
                for i in self.matdata:
                    if i == "__version__" or i == "__header__" or i == "__globals__":
                        continue
                    elif i=="angles":
                        self.matangles = self.matdata[i]
                    else:
                        self.matsize = self.matdata[i].shape
                        self.listData.addItem(i)
                        self.matkeys.append(i)
            print("matsize: ", self.matsize)

    def __load_model(self):
        # if not self.ori_image:
        #     raise Exception("No image is selected.")
        self.img_num.setMaximum(self.matsize[1]-1)
        self.img_num_show.setRange(0,self.matsize[1]-1)
        self.cuda = self.use_cuda.isChecked()
        model_path = os.path.join(self.__curdir, self.__models[self.model_name])
        recon_path = self.recon_name
        self.ori_content = self.ori_image
        dirname = os.path.dirname(self.imagePath_content)
        dataname = self.listData.currentText()
        result_d, result = load_model(dataname, self.matdata, self.matangles, self.matsize, model_path,\
                                 recon_path, self.cuda, \
                                 self.sart_iter.value(), self.sirt_iter.value(), self.set_iter.value())
        savename = os.path.join(dirname, dataname)
        if not os.path.exists(savename):
            os.mkdir(savename)
        scio.savemat(savename + "/angles.mat", {"angles":self.matangles})
        if self.model_name=="Deepfusion(3d)" or self.model_name=="Deepfusion(3dGAN)":
            scio.savemat(savename + "/" + "%s_%s.mat" % (dataname,recon_path), result)
            scio.savemat(savename + "/" + "%s_%s.mat" % (dataname,self.model_name), result_d)
        else:
            scio.savemat(savename + "/" + "%s_%s.mat" % (dataname,recon_path), result)
            scio.savemat(savename + "/" + "%s_%s_%s.mat" % (dataname, self.model_name, recon_path), result_d)

        print("Save mat successfully!")
        self.model_out = result_d
        self.recon_out = result
        dataname = self.listData.currentText()
        idx = self.img_num.value()
        print(dataname, idx)
        model_out = self.model_out[dataname][idx, :, :]
        model_out = (model_out * 255).astype('uint8')
        model_out = Image.fromarray((model_out), mode='L')
        model_out = PIL2Pixmap(model_out)
        model_out.scaled(self.model_output.size(), QtCore.Qt.KeepAspectRatio)
        self.model_output.setPixmap(model_out)
        self.model_output.show()
        recon_out = map01(self.recon_out[dataname][idx, :, :])
        recon_out = (recon_out * 255  / np.max(recon_out)).astype('uint8')
        recon_out = Image.fromarray((recon_out), mode='L')
        recon_out = PIL2Pixmap(recon_out)
        recon_out.scaled(self.model_output.size(), QtCore.Qt.KeepAspectRatio)
        self.recon_output.setPixmap(recon_out)
        self.recon_output.show()

    def ArraytoBinary(self):
        import struct
        dirname = os.path.dirname(self.imagePath_content)
        path = os.path.join(dirname, dirname, "binary_files")
        if not os.path.exists(path):
            os.mkdir(path)
        for m in self.model_out:
            array = self.model_out[m]
            with open(path + "/%s_%s_float32_%s_%s.bin" % \
                      (m,array.shape,self.recon_name, self.model_name), 'wb')as fp:
                for i in range(len(array[0][0])):  #
                    for j in range(len(array[0])):  #
                        for k in range(len(array)):  #
                            dataStr = struct.pack('f', array[k][j][i])
                            fp.write(dataStr)

        for n in self.recon_out:
            array = self.recon_out[n]
            with open(path + "/%s_%s_float32_%s.bin" % \
                      (n, array.shape, self.recon_name), 'wb')as fp:
                for i in range(len(array[0][0])):  #
                    for j in range(len(array[0])):  #
                        for k in range(len(array)):  #
                            dataStr = struct.pack('f', array[k][j][i])
                            fp.write(dataStr)
        print("Convert to Binary files Successfully!!")


    def LoadModel(self):
        self.model_name = self.modelPath.currentText()
        self.recon_name = self.reconPath.currentText()
        self.show_flag = False
        self.img_num.setValue(0)
        # if not self.ori_image:
        #     QMessageBox.warning(self, "You need select an image!", self.tr("You need select an image!"))
        #     return
        self.__load_model()
        self.show_flag = True
        # self.Denoise()



    def ResultShow(self):
        dataname = self.listData.currentText()
        idx = self.img_num.value()
        # print(self.img_num.value())
        self.img_num_show.setValue(self.img_num.value())
        # print(dataname, idx)
        if self.show_flag:
            model_out = self.model_out[dataname][idx,:,:]
            model_out = (model_out * 255).astype('uint8')
            model_out = Image.fromarray((model_out), mode='L')
            model_out = PIL2Pixmap(model_out)
            model_out.scaled(self.model_output.size(), QtCore.Qt.KeepAspectRatio)
            self.model_output.setPixmap(model_out)
            self.model_output.show()

            recon_out = map01(self.recon_out[dataname][idx, :, :])
            recon_out = (recon_out * 255 / np.max(recon_out)).astype('uint8')
            recon_out = Image.fromarray((recon_out), mode='L')
            recon_out = PIL2Pixmap(recon_out)
            recon_out.scaled(self.model_output.size(), QtCore.Qt.KeepAspectRatio)
            self.recon_output.setPixmap(recon_out)
            self.recon_output.show()




    def drawPoint(self, event):
        self.pos = event.pos()
        self.update()

    def release(self):
        self.model_output.clear()
        self.img_num.setValue(0)
        return

    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self,
                                                "Confirm Exit...",
                                                "Are you sure you want to exit?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        event.ignore()

        if result == QtWidgets.QMessageBox.Yes:
            self.release()
            event.accept()


qtCreatorFile = os.path.join("UI_files", "WinUI.ui")

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Code_MainWindow()
    window.show()
    sys.exit(app.exec_())
