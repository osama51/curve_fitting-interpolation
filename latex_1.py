import os
import sys
from os import path
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
from math import ceil, floor

FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self , parent=None):
        pg.setConfigOption('background', (0,0,0))
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.flag = 0
        self.pen=pg.mkPen(color='c')
        self.x=[2.3,3.5,4.6,5.0,6.1,7.0,7.2,8.5,8.6,8.8,8.9, 9, 9.1, 9.6, 10., 10.5]
        self.signal=[1.2,2.6,3.1,4.2,5.3,5.8,6.4,7.0,7.6,8.0,8.5, 8.7, 9, 9.5, 10.1, 10.6]
        self.length_original_data = len(self.signal)
        self.actionSamples.triggered.connect(self.show_sample_alone)
        self.actionPlot.triggered.connect(self.draw_signal)
        self.pushButton.pressed.connect(self.error)
        

    def mathTex_to_QPixmap(self):
        # mathTex = [
        # '$C_{soil}=(1 - n) C_m + theta_w C_w$',
        # '$k_{soil}=frac{sum f_j k_j theta_j}{sum f_j theta_j}$',
        # '$lambda_{soil}=k_{soil}  C_{soil}$']
        mathTex = 'f(x)=' +self.ystring 
        #---- set up a mpl figure instance ----
    
        fig = mpl.figure.Figure()
        fig.patch.set_facecolor('none')
        fig.set_canvas(FigureCanvasAgg(fig))
        renderer = fig.canvas.get_renderer()
    
        #---- plot the mathTex expression ----
    
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_facecolor('none')
        t = ax.text(0, 0, mathTex, ha='left', va='bottom', fontsize=22)
    
        #---- fit figure size to text artist ----
    
        fwidth, fheight = fig.get_size_inches()
        fig_bbox = fig.get_window_extent(renderer)
    
        text_bbox = t.get_window_extent(renderer)
    
        tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
        tight_fheight = text_bbox.height * fheight / fig_bbox.height
    
        fig.set_size_inches(tight_fwidth, tight_fheight)
    
        #---- convert mpl figure to QPixmap ----
    
        buf, size = fig.canvas.print_to_buffer()
        qimage = QtGui.QImage.rgbSwapped(QtGui.QImage(buf, size[0], size[1],
                                                      QtGui.QImage.Format_ARGB32))
        qpixmap = QtGui.QPixmap(qimage)
        self.label.setPixmap(qpixmap)

            
    def show_sample_alone(self):
        if self.actionSamples.isChecked():
            self.graphicsView.plotItem.clearPlots()
            self.graphicsView.plot(self.x, self.signal,
                  pen=None,
                  name="BEP",
                  symbol='o',
                  symbolPen=pg.mkPen(color=(255, 0, 0), width=0),                                      
                  symbolBrush=pg.mkBrush(255, 255, 0, 255),
                  symbolSize=7)
        else :
            self.graphicsView.plotItem.clearPlots()
            
    def devide_to_chunks(self,length_of_data, number_of_chunks, overlap_percentage):
        overlap_percentage /= 100
        self.indices_list = [0]
        ideal_length_of_chunk = ceil(length_of_data/number_of_chunks)
        length_of_chunk = ceil(length_of_data/(number_of_chunks-((number_of_chunks-1)*(overlap_percentage))))
        
        for index in range(0,2*(number_of_chunks-1),2):
            # normal_index=int(self.indices_list[index])+length_of_chunk
            # margin=ceil((overlap_percentage/2)*ideal_length_of_chunk)
            index_after=self.indices_list[index]+length_of_chunk
            index_before=ceil(index_after-(overlap_percentage*length_of_chunk))
            
            if (index_after < length_of_data) and (index_before < length_of_data):
                self.indices_list.append(index_after)
                self.indices_list.append(index_before)
        self.indices_list.append(length_of_data-1)
        

    def error(self):
        if not self.flag:
            self.pushButton.setText('Cancel')
            self.flag = 1

            chunks= np.arange(1,4)
            overlaps = np.arange(0,21, 5)
            orders = np.arange(3,5)
            # chunks = np.arange(1,5)
            # overlaps = np.arange()
            
            # print(orders,'order')
            self.Error_map = np.zeros((len(chunks),len(overlaps),len(orders)))
            for chunk in range(len(chunks)):
                for overlap in range(len(overlaps)):
                    self.devide_to_chunks(len(self.signal),chunks[chunk],overlaps[overlap])
                    print(self.indices_list)
                    print(overlaps[overlap], 'overlap', chunks[chunk], 'chunk')
                    for index in range(0,len(self.indices_list),2):
                        for order in orders:
                            # print(index,'in')
                            print(order, 'order')
                            self.Error_map[chunk,overlap,order-3]=self.interpolation([index,index+1],order)
                            print(self.Error_map[chunk,overlap,order-3], 'ERRRRR')
                            
                            # print(order,'ininter')
            # print(Error_map[5,3,2])
        else:
            self.pushButton.setText('Start')
            self.flag = 0
        
    def draw_signal(self):
        self.graphicsView.plot(self.x, self.signal, pen = self.pen, title="Three plot curves")
        self.devide_to_chunks(16,1,0)
        self.interpolation([0,1],10)
        self.draw_interpolation()
        # self.draw_spectrogram()
        self.mathTex_to_QPixmap()
        if self.actionSamples.isChecked():
            self.self.error()


    def interpolation(self,indices_list,order):
        print(indices_list, 'indices', self.indices_list, 'seldindie')
        kkk = indices_list[0]
        index1 = self.indices_list[kkk]
        print(indices_list[1], ' indices1')
        index2 = self.indices_list[indices_list[1]]+1
        y=np.array(self.signal[index1:index2])
        # print(y,'y')
        print(self.signal[self.indices_list[indices_list[0]]:self.indices_list[indices_list[1]]])
        x=np.array(self.x)
        x=np.array(x[self.indices_list[indices_list[0]]:self.indices_list[indices_list[1]]+1])
        num_of_samples = len(y)
        # print(self.length_original_data,'lenn')
        step = ceil((self.length_original_data/order)-1)
        if step > 3: 
            return 0
            
        print(step, 'step')
        print(len(y), 'yyy')
        y_samples = []
        x_samples = []
        # for counter in range(0,num_of_samples,step):
        #     y_samples.append(y[counter])
        #     x_samples.append(x[counter])
        y_samples = y[0:num_of_samples+1:step]
        x_samples = x[0:num_of_samples+1:step]
        n = int(len(x_samples))

        print(x_samples, 'x samples', y_samples, 'y_samples')
        if len(y_samples)!=n:print('x and y must be same length')
        b = np.zeros((n,n),dtype=float)
        # print('ysss',y_samples[:])
        print(len(b), 'shapeb')
        b[:,1] = y_samples[:]
        
        for m in range(2,n+1):
            for i in range(0,n-m): 
                b[i,m] = (b[i+1,m-1]-b[i,m-1])/(x_samples[i+m-1]-x_samples[i])
                
        xx=np.arange(2,11.,.1)
        np.around(xx, 2, xx)
        # print(xx[3],'hatoly el kalb dah')
        xt = 1
        # yint = np.zeros(len(xx))
        yint = b[1,1]
        ystring =str(b[1,1])
        xt_str =""
        # xt_str += f"(x-{x_samples[m]})"
        # ystring += f"+({b[1,m+1]})({xt_str})"
        coeff = b[1]
        np.around(coeff, 2, coeff)
        
        for m in range( 1,n-1):
            xt = xt*(xx-x_samples[m])
            yint = yint+b[1,m+1]*xt
            if order > 4 and m > 2 and m < n-5:
                ystring += "."
            else:
                xt_str += f"(x-{x_samples[m]})"
                new_coeff = coeff[m+1]
                if new_coeff < 0.1 and new_coeff > - 0.1 :continue
                ystring += f"+({new_coeff})({xt_str})"
        self.ystring = ystring
        # print(ystring,'yint')
        if step == 2:
            plt.plot(xx,yint)
            plt.show()
        xx = list(xx)
        self.yint = yint
        self.xx = xx
        yint_picked = [] 
        for z in x:
            yint_picked.append(yint[xx.index(z)])
        result = map(lambda original_data, interpolated_data: abs((interpolated_data - original_data)/original_data), y, yint_picked)
        result = list(result)
        error = sum(result[1:-1])/len(result[1:-1])
        return error 
    
    def draw_interpolation(self):
        self.graphicsView.plot(self.xx, self.yint, pen ='g')
        
    def portion(self,data_lenght,portion_percentage):
        self.indices_list =[0]
        last_index = int(((portion_percentage/100)*data_lenght)-1)
        self.indices_list.append(last_index)
        
    def draw_spectrogram(self):
        self.graphicsView.clear()
        
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.mkQApp()
        spec_plot = self.graphicsView.addPlot()
        img = pg.ImageItem()
        spec_plot.addItem(img)

        # self.grSpec.show()
        img.setImage(self.Error_map[1]) # Sxx: amplitude for each pixel
        img.scale(t[-1]/np.size(Sxx, axis=1),
                  f[-1]/np.size(Sxx, axis=0))
        
        spec_plot.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])
        spec_plot.setLabel('bottom', "Time", units='s')
        spec_plot.setLabel('left', "Frequency", units='Hz')
        
    
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
    # def on_latex(self):
    # expr = "$\displaystyle " + str(self.x) + "$"
        # # expr = str(self.x)
        # print(str(self.x))
        # #This creates a ByteIO stream and saves there the output of sympy.preview
        # f = BytesIO()
        # #  preamble = "\\documentclass[10pt]{article}\n" \
        # # "\\usepackage{amsmath,amsfonts}\\begin{document}"
        # # the_color = "{" + self.master.cget('bg')[1:].upper()+"}"
        # sp.preview(expr,
        # viewer = "BytesIO", output = "png", outputbuffer=f)
        # f.seek(0)
        # # print (f.seek(0))
        # #Open the image as if it were a file. This works only for .ps!
        # img = Image.open(f)
        # # img = f
        # #See note at the bottom
        # # img.load(scale = 10)
        # # img = img.resize((int(img.size[0]/2),int(img.size[1]/2)),Image.BILINEAR)
        # qimage = ImageQt(img)
        # pixmap = QtGui.QPixmap.fromImage(qimage)
        # self.label.setPixmap(qpixmap)
        # self.label.resize(10,10)
        # photo = ImageTk.PhotoImage(img)
        # self.label.config(image = photo)
        # self.label.image = photo
        # # f.close()