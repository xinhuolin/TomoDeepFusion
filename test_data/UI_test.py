#QSplider（滑动条）控件的使用
from PyQt5.QtWidgets import  QVBoxLayout,QWidget,QApplication ,QHBoxLayout,QSpinBox,QSlider,QLabel

from PyQt5.QtGui import QIcon,QPixmap,QFont
from PyQt5.QtCore import  Qt

import sys

class WindowClass(QWidget):

    def __init__(self,parent=None):

        super(WindowClass, self).__init__(parent)
        layout=QVBoxLayout()
        self.label_0 = QLabel()
        self.label_0.setText("文本字体大小为：")

        self.label=QLabel()
        self.label.setFont(QFont(None,20))

        self.splider=QSlider(Qt.Horizontal)
        self.splider.valueChanged.connect(self.valChange)
        self.splider.setMinimum(20)#最小值
        self.splider.setMaximum(60)#最大值
        self.splider.setSingleStep(2)#步长
        self.splider.setTickPosition(QSlider.TicksBelow)#设置刻度位置，在下方
        self.splider.setTickInterval(5)#设置刻度间隔

        layout.addWidget(self.splider)
        layout.addWidget(self.label_0)
        layout.addWidget(self.label)
        self.resize(500,500)
        self.setLayout(layout)

    def valChange(self):
        print(self.splider.value())
        self.label.setNum(self.splider.value())#注意这里别setText 会卡死
        self.label_0.setFont(QFont("微软雅黑",self.splider.value()))

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=WindowClass()
    win.show()
    sys.exit(app.exec_())