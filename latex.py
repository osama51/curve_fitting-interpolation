import os
import sys
from os import path
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUiType
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer
import _thread 
import pyqtgraph as pg
from tkinter import *
import sympy as sp
from PIL import Image
from PIL.ImageQt import ImageQt
from io import BytesIO
from sympy import symbols, preview, Symbol


FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self , parent=None):
        pg.setConfigOption('background', (0,0,0))
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.flag = 0
        self.pen=pg.mkPen(color='c')
        self.x=[2.3,3.5,4.6,5.0,6.1,7.0,7.2,8.5,8.6,8.8,8.9]
        self.y=[1.2,2.6,3.1,4.2,5.3,5.8,6.4,7.0,7.6,8.0,8.5]
        
        self.actionSamples.triggered.connect(self.show_sample_alone)
        self.actionPlot.triggered.connect(self.draw_signal)
        self.pushButton.pressed.connect(self.error)
        
    def on_latex(self):
            x, y = symbols("x,y")
            # expr = "$\displaystyle " + str(self.x) + "$"
            # expr = str(self.x)
            print(str(self.x))
            #This creates a ByteIO stream and saves there the output of sympy.preview
            f = BytesIO()
           #  preamble = "\\documentclass[10pt]{article}\n" \
           # "\\usepackage{amsmath,amsfonts}\\begin{document}"
           
            # the_color = "{" + self.master.cget('bg')[1:].upper()+"}"
            sp.preview(x + y,  
                       viewer = "BytesIO", output = "png", outputbuffer=f)
            f.seek(0)
            print (f.seek(0))
            #Open the image as if it were a file. This works only for .ps!
            img = Image.open(f)
            # img = f
            #See note at the bottom
            # img.load(scale = 10)
            # img = img.resize((int(img.size[0]/2),int(img.size[1]/2)),Image.BILINEAR)
            qimage = ImageQt(img)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.label.setPixmap(pixmap)
            # self.label.resize(10,10)
            # photo = ImageTk.PhotoImage(img)
            # self.label.config(image = photo)
            # self.label.image = photo
            f.close()
            
    def show_sample_alone(self):
        if self.actionSamples.isChecked():
            self.graphicsView.plotItem.clearPlots()
            self.graphicsView.plot(self.x, self.y,
                  pen=None,
                  name="BEP",
                  symbol='o',
                  symbolPen=pg.mkPen(color=(255, 0, 0), width=0),                                      
                  symbolBrush=pg.mkBrush(255, 255, 0, 255),
                  symbolSize=7)
        else :
            self.graphicsView.plotItem.clearPlots()
            
    def error(self):
        if not self.flag:
            self.pushButton.setText('Cancel')
            self.flag = 1
        else:
            self.pushButton.setText('Start')
            self.flag = 0
        
    def draw_signal(self):
        self.on_latex()
        self.graphicsView.plot(self.x, self.y, pen = self.pen)
        if self.actionSamples.isChecked():
            self.self.error()
            
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()
    
    
    # sp.preview(expr, euler = False, preamble = r"\documentclass{standalone}"
    #                    r"\usepackage{pagecolor}"
    #                    r"\definecolor{graybg}{HTML}" + the_color +
    #                    r"\pagecolor{graybg}"
    #                    r"\begin{document}",
    #                    viewer = "BytesIO", output = "ps", outputbuffer=f)