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
import numpy.polynomial.polynomial as poly
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self , parent=None):
        pg.setConfigOption('background', (0,0,0))
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.flag = 0
        self.btengan = 0
        self.pen=pg.mkPen(color='c')
        # self.x=[2.3,3.5,4.6,5.0,6.1,7.0,7.2,8.5,8.6,8.8,8.9, 9, 9.1, 9.6, 10., 10.5]
        # self.signal=[1.2,2.6,3.1,4.2,5.3,5.8,6.4,7.0,7.6,8.0,8.5, 8.7, 9, 9.5, 10.1, 10.6]
        # self.length_original_data = len(self.signal)
        self.actionSamples.triggered.connect(self.show_sample_alone)
        self.actionPlot.triggered.connect(self.draw_signal)
        self.pushButton.pressed.connect(self.error)
        self.actionOpen.triggered.connect(self.prepare_data)

        self.portion_horizontalSlider.valueChanged.connect(self.handle_sliders)
        self.order_horizontalSlider.valueChanged.connect(self.handle_sliders)
        self.order_horizontalSlider.valueChanged.connect(lambda:self.handle_sliders)
        self.chunks_horizontalSlider.valueChanged.connect(self.handle_sliders)
        self.overlap_horizontalSlider.valueChanged.connect(self.handle_sliders)
        self.chunks_horizontalSlider.valueChanged.connect(lambda:self.devide_to_chunks(int(self.length_original_data)
            ,self.chunks_horizontalSlider.value(),self.overlap_horizontalSlider.value()))
        self.overlap_horizontalSlider.valueChanged.connect(lambda:self.devide_to_chunks(int(self.length_original_data)
            ,self.chunks_horizontalSlider.value(),self.overlap_horizontalSlider.value()))
        
    def handle_sliders(self):
        self.porrtion_label.setText(str(self.portion_horizontalSlider.value()))
        self.order_label.setText(str(self.order_horizontalSlider.value()))
        self.chunks_label.setText(str(self.chunks_horizontalSlider.value()))
        self.overlap_label.setText(str(self.overlap_horizontalSlider.value()))
        
    def mathTex_to_QPixmap(self):
        # mathTex = [
        # '$C_{soil}=(1 - n) C_m + theta_w C_w$',
        # '$k_{soil}=frac{sum f_j k_j theta_j}{sum f_j theta_j}$',
        # '$lambda_{soil}=k_{soil}  C_{soil}$']
        mathTex = 'f(x)=' #+self.ystring 
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
                  symbol='x',
                  symbolPen=pg.mkPen(color=(255, 0, 0), width=0),                                      
                  symbolBrush=pg.mkBrush(255, 255, 0, 255),
                  symbolSize=7)
        else :
            self.graphicsView.plotItem.clearPlots()
            
    def devide_to_chunks(self,length_of_data, number_of_chunks, overlap_percentage):
        overlap_percentage /= 100
        self.indices_list = [0]
        # ideal_length_of_chunk = ceil(length_of_data/number_of_chunks)
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
        

    def draw_interpolation(self):
        # print(len(self.y_int))
        # self.graphicsView.plot(self.x2, self.y_int, pen ='g')
        self.draw_chunks(5)
       
    def draw_signal(self):
        self.graphicsView.plot(self.x, self.signal, pen = self.pen, title="Three plot curves")
        self.devide_to_chunks(1000,5,20)
        # self.interpolation([0,1],1)
        # self.interpolation_builtin([0,1],1)
        self.draw_interpolation()
        self.mathTex_to_QPixmap()
        if self.actionSamples.isChecked():
            self.error()

    def error(self):
        if not self.flag:
            self.pushButton.setText('Cancel')
            self.flag = 1

            chunks= np.arange(1,4)
            overlaps = np.arange(0,21, 5)
            orders = np.arange(3,5)
  
            self.btengan = 0
            summ = 0
            self.Error_map = np.zeros((len(chunks),len(overlaps),len(orders)))
            for chunk in range(len(chunks)-1):
                for overlap in range(len(overlaps)-1):
                    self.devide_to_chunks(len(self.signal),chunks[chunk],overlaps[overlap])
                    print(self.indices_list)
                    print(overlaps[overlap], 'overlap', chunks[chunk], 'chunk')
                    for index in range(0,len(self.indices_list),2):
                        if self.btengan == chunks[chunk]:
                            self.btengan = 0
                            self.Error_map[chunk,overlap,order-3] = summ/chunks[chunk]
                        for order in orders:
                            summ += self.interpolation([index,index+1],order)
                            self.btengan += 1
                            print(order, 'order')
                            # self.Error_map[chunk,overlap,order-3]=self.interpolation([index,index+1],order)
                            print(self.Error_map[chunk,overlap,order-3], 'ERRRRR')
                            print(self.Error_map[:,:,0],'slaysaia')
        else:
            self.pushButton.setText('Start')
            self.flag = 0
    
    def draw_chunks(self, chunks):
        y_int = np.zeros(len(self.signal))
        i = 1
        for chunk in range(0,chunks,1):
            self.interpolation([chunk+i-1,chunk+i],5)
            y_int[self.index1:self.index2] += self.y_int
            i += 1
        m = 0
        for h in range(1,chunks,1):
            y_int[self.indices_list[2*h]:self.indices_list[2*h-1]+1] /= 2
            print(self.indices_list[2*h],self.indices_list[2*h-1])
        x2= np.arange(0,len(y_int),1)
        self.graphicsView.plot(x2, y_int, pen='g')
        
    def interpolation_builtin(self, x, y, degree):

        #def calculate_coff(x,y,degree):
        y=np.array(y)
        x=np.array(x)
        
        # print(coff)
        x2= np.arange(0,len(y),1)
        y_int=np.array([])
        # print(np.shape(y_int))
        
        # tck = interpolate.splrep(x, y, s=0)
        # ynew = interpolate.splev(x2, tck, der=0)
        # print('x2', x2, 'y', y)
        spl = UnivariateSpline(x2, y, k= degree)
        
        # print(spl(x2))
        self.x2 = x2 
        self.y_int = spl(x2)
        # print(y_int)
        # plt.plot(x2,y_int,'b')
        # plt.show()
        
    def interpolation(self,indices_list,order):
        
        self.length_original_data = len(self.signal)
        print(indices_list, 'indices', self.indices_list, 'selfindie')
        first_ind = indices_list[0]
        self.index1 = self.indices_list[first_ind]
        # print(indices_list[1], ' indices1')
        self.index2 = self.indices_list[indices_list[1]]+1
        y=np.array(self.signal[self.index1:self.index2])

        x=np.array(self.x)
        x=np.array(x[self.indices_list[indices_list[0]]:self.indices_list[indices_list[1]]+1])
        num_of_samples = len(y)

        self.interpolation_builtin(x, y, order)
        # step = ceil((self.length_original_data/order)-1)
        # # if step > 3: 
        # #     return 0
        # # print(step, 'step')
        # # print(len(y), 'yyy')
        # y_samples = []
        # x_samples = []

        # y_samples = y[0:num_of_samples+1:step]
        # x_samples = x[0:num_of_samples+1:step]
        # n = int(len(x_samples))

        # # print(x_samples, 'x samples', y_samples, 'y_samples')
        # if len(y_samples)!=n:print('x and y must be same length')
        # b = np.zeros((n,n),dtype=float)
        # # print('ysss',y_samples[:])
        # # print(len(b), 'shapeb')
        # b[:,1] = y_samples[:]
        # # print(b, 'bbbbb')
        # for m in range(2,n+1):
        #     for i in range(0,n-m): 
        #         b[i,m] = (b[i+1,m-1]-b[i,m-1])/(x_samples[i+m-1]-x_samples[i])
                
        self.xx =np.arange(self.x[0],len(self.x),1)
        np.around(self.xx, 2, self.xx)
        # # print(self.xx[3],'hatoly el kalb dah')
        # xt = 1
        # # yint = np.zeros(len(self.xx))
        # yint = b[1,1]
        # ystring =str(b[1,1])
        # xt_str =""
        # coeff = b[1]
        # np.around(coeff, 2, coeff)
        # for m in range( 0,n-1):
        #     xt = xt*(self.xx-x_samples[m]) 
        #     yint = yint+b[1,m+1]*xt
        #     # print(type(xt), 'ana xt')
        #     # if order > 4 and m > 2 and m < n-3:
        #     #     ystring += "."
        #     # else:
        #     xt_str += f"(x-{x_samples[m]})"
        #     new_coeff = coeff[m+1]
        #     if new_coeff < 0.1 and new_coeff > - 0.1 :continue
        #     ystring += f"+({new_coeff})({xt_str})"
        # self.ystring = ystring
        # # print( type(yint), 'two')
        # # print(b, 'bbbbbeeeb')
        # if step == 2:
        #     plt.plot(self.xx,yint)
        #     plt.show()
        self.xx = list(self.xx)
        self.yint = self.y_int
        yint_picked = [] 

        # for z in x:
        #     print(z, ' zzz')
        #     index = self.xx.index(z)
        #     print(index, 'index')
        #     print(self.yint[index-1])
        #     yint_picked.append(self.yint[index])
        # result = map(lambda original_data, interpolated_data: abs((interpolated_data - original_data)/original_data), y, yint_picked)
        # result = list(result)
        # error = sum(result[1:-1])/len(result[1:-1])
        # return error 
    
        
    def portion(self,data_lenght,portion_percentage):
        self.indices_list =[0]
        last_index = int(((portion_percentage/100)*data_lenght)-1)
        self.indices_list.append(last_index)

    #_____________________________FETCHING DATA________________________#
    
    def prepare_data(self):
        self.browse()
        self.read_data()
        self.x = self.timestamps
        self.signal = self.amplitude
        print(len(self.x))
        
    def browse(self):
        self.file_path_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', " ", "(*.txt *.csv *.xls)")

    def read_data(self):
        self.file_name, self.file_extension = os.path.splitext(self.file_path_name)
        
        if self.file_extension == '.txt':
            # to skip the second row, use a list-like argument "[]"
            # fwf stands for fixed width formatted lines.
            data = pd.read_fwf(self.file_path_name)
            self.signal_name = data.columns[1]
            self.amplitude = data.iloc[:, 1] # gets second column
            # : means everything in dimension1 from the beginning to the end
            self.timestamps = data.iloc[:, 0]
            self.interval = self.timestamps[6]- self.timestamps[5]
            
        elif self.file_extension == '.csv':
            data = pd.read_csv(self.file_path_name)
            self.signal_name = data.columns[1]
            self.amplitude = data.iloc[:, 1] # gets second column
            self.timestamps = data.iloc[:, 0]
            self.interval = self.timestamps[6]- self.timestamps[5]
        self.sampling_freq = 1000#1/interval
        # xy_axes = {'xaxis': xaxis_timestamps, 'yaxis': yaxis_values}
        # return xy_axes;

    
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()
    
# def interpolation_builtin(self, indices_list, degree):

#         #def calculate_coff(x,y,degree):
#         y=np.array(self.signal)
#         x=np.array(self.x)
#         coff = poly.polyfit(x,y,degree)

#         # print(coff)
#         x2= np.arange(0,len(y),1)
#         y_int=np.array([])
#         # print(np.shape(y_int))
#         self.x2 = x2 
        
#         for i in range(len(coff)):
#             if np.shape(y_int) == (0,):
#                 y_int = coff[0]*x2
#             else:
#                 y_int += coff[i]*x2**i
                
#         self.y_int = y_int
#         # print(y_int)
#         plt.plot(x,y,'r',x2,y_int,'b')
#         plt.show()