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
# from PyQt5.QtCore import QTimer
import _thread 
import pyqtgraph as pg
# from PIL import Image
# from PIL.ImageQt import ImageQt
# from io import BytesIO
from math import ceil, floor
from threading import Thread

FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self , parent=None):
        pg.setConfigOption('background', (0,0,0))
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        # self.pg = self.graphicsView.addPlot()
        # self.pg.addLegend()
        self.flag = 0
        self.btengan = 0
        self.plotted = 0
        self.start_error = 1
        self.formulae = []
        self.pen=pg.mkPen(color='c')
        # self.x=[2.3,3.5,4.6,5.0,6.1,7.0,7.2,8.5,8.6,8.8,8.9, 9, 9.1, 9.6, 10., 10.5]
        # self.signal=[1.2,2.6,3.1,4.2,5.3,5.8,6.4,7.0,7.6,8.0,8.5, 8.7, 9, 9.5, 10.1, 10.6]
        
        
        self.actionSamples.triggered.connect(self.show_samples)
        self.actionPlot.triggered.connect(self.draw_signal)
        self.actionOpen.triggered.connect(self.prepare_data)
        self.actionstring.triggered.connect(self.controlling_error_axes)
        self.starterrorButton.pressed.connect(self.error_thread)
        self.order_horizontalSlider.valueChanged.connect(self.run_sliders)
        self.chunks_horizontalSlider.valueChanged.connect(self.run_sliders)
        self.overlap_horizontalSlider.valueChanged.connect(self.run_sliders)

        self.portion_horizontalSlider.valueChanged.connect(self.handle_sliders)
        self.order_horizontalSlider.valueChanged.connect(self.handle_sliders)
        self.order_horizontalSlider.valueChanged.connect(lambda:self.handle_sliders)
        self.chunks_horizontalSlider.valueChanged.connect(self.handle_sliders)
        self.overlap_horizontalSlider.valueChanged.connect(self.handle_sliders)
        self.chunks_horizontalSlider.valueChanged.connect(lambda:self.devide_to_chunks(int(self.length_original_data)
            ,self.chunks_horizontalSlider.value(),self.overlap_horizontalSlider.value()))
        self.overlap_horizontalSlider.valueChanged.connect(lambda:self.devide_to_chunks(int(self.length_original_data)
            ,self.chunks_horizontalSlider.value(),self.overlap_horizontalSlider.value()))
        
    def run_sliders(self):
        if self.actionPlot.isChecked():
            self.draw_signal()
            
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
        np.around(self.coeff, 3, self.coeff)
  
        terms = [self.coeff[0]]
        terms = str(terms)
        for i in range(1,self.degree+1,1):
            if self.coeff[i] < 0.01 and self.coeff[i] > - 0.01 :continue
            print(i, 'i', self.coeff[i])
            # coeff = self.coeff[i]
            terms += f'$+ ({self.coeff[i]}) x^{i} $'
            # terms.append(expr)
                
        print(terms)
        mathTex = 'f(x)=' + terms
        print(mathTex)
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
        # self.formulae += qimage
        qpixmap = QtGui.QPixmap(qimage)
        label = QtGui.QLabel(pixmap=qpixmap)
        # self.scrollArea.setWidgetResizable(False)
        self.formulaeLayout.addWidget(label)
        # self.label.setPixmap(qpixmap)
            
    def show_samples(self):
        if self.actionSamples.isChecked():
            
            self.graphicsView.plot(self.x_axis, self.y_axis,
                  pen=None,
                  name="BEP",
                  symbol='x',
                  symbolPen=pg.mkPen(color=(255, 0, 0), width=0),                                      
                  symbolBrush=pg.mkBrush(255, 255, 0, 255),
                  symbolSize=7)
            
        else :
            if not self.actionSamples.isChecked():
                if self.plotted:
                    self.graphicsView.plotItem.clearPlots()
                    self.draw_signal()
                else:
                    self.graphicsView.plotItem.clearPlots()
                    
    def error_thread(self):
        if self.start_error:
            # t = Thread(target = self.draw_errormap,)
            _thread.start_new_thread(self.draw_errormap,())
            # t.start()
            self.start_error = 0
            print(self.start_error)
        else:
            _thread.exit()
            self.start_error = 1
        self.error_button_text()
        
    def controlling_error_axes(self):
        # self.Error_map = np.zeros((len(chunks),len(overlaps),len(orders)))
        condition1 = self.order_x_radioButton.isChecked() and self.order_y_radioButton.isChecked()
        condition2 = self.chunks_x_radioButton.isChecked() and self.chunks_y_radioButton.isChecked()
        condition3 = self.overlap_x_radioButton.isChecked() and self.overlap_y_radioButton.isChecked()
        if (condition1) or (condition2) or (condition3):
            self.label_alert.setText("Pick different variables for the same graph!😑")
        else:
            self.label_alert.setText("😘😘")
            
        if self.order_x_radioButton.isChecked() and self.chunks_y_radioButton.isChecked():
            self.img_map=self.Error_map[:,1,:]
        elif self.order_x_radioButton.isChecked() and self.overlap_y_radioButton.isChecked():
            self.img_map=self.Error_map[1,:,:]
        elif self.chunks_x_radioButton.isChecked() and self.order_y_radioButton.isChecked():
            self.img_map=np.transpose(self.Error_map[:,1,:])
        elif self.chunks_x_radioButton.isChecked() and self.overlap_y_radioButton.isChecked():
            self.img_map=self.Error_map[:,:,1]
        elif self.overlap_x_radioButton.isChecked() and self.order_y_radioButton.isChecked():
            self.img_map=np.transpose(self.Error_map[1,:,:])
        elif self.overlap_x_radioButton.isChecked() and self.chunks_y_radioButton.isChecked():
            self.img_map=np.transpose(self.Error_map[:,:,1])
            
    def devide_to_chunks(self,length_of_data, number_of_chunks, overlap_percentage):
        overlap_percentage /= 100
        self.indices_list = [0]
        length_of_chunk = ceil(length_of_data/(number_of_chunks-((number_of_chunks-1)*(overlap_percentage))))
        
        for index in range(0,2*(number_of_chunks-1),2):

            index_after=self.indices_list[index]+length_of_chunk
            index_before=ceil(index_after-(overlap_percentage*length_of_chunk))
            
            if (index_after < length_of_data) and (index_before < length_of_data):
                self.indices_list.append(index_after)
                self.indices_list.append(index_before)
        self.indices_list.append(length_of_data-1)
        
    def interpolation(self, indices_list, degree):
        print(indices_list, 'indices', self.indices_list, 'selfindie')
        first_ind = indices_list[0]
        self.index1 = self.indices_list[first_ind]
        self.index2 = self.indices_list[indices_list[1]]+1
        y=np.array(self.y_axis[self.index1:self.index2])
        x=np.array(self.x_axis[self.indices_list[indices_list[0]]:self.indices_list[indices_list[1]]+1])
        self.piece_interpolation(x, y, degree)

    def piece_interpolation(self, x_data, y_data, degree):
        self.coeff = np.polyfit(x_data, y_data,degree)
        self.y_fitted = np.polyval(self.coeff, x_data)
        error = (self.y_fitted - y_data)/y_data
        error = np.abs(error)
        error = sum(error[1:])/len(error)
        return error
    
    def error_map(self):
        chunks= np.arange(1,21)
        overlaps = np.arange(0,25, 6)
        orders = np.arange(1,50)
        progressbar_index = 0
        progressbar_percentage = 0
        total_percentage = len(chunks)*len(overlaps)*len(orders)
        # indices_list = 
        btengan = 0
        summ = 0
        self.Error_map = np.zeros((len(chunks),len(overlaps),len(orders)))
        for chunk in range(len(chunks)):
            for overlap in range(len(overlaps)-1):
                self.devide_to_chunks(len(self.y_axis),chunks[chunk],overlaps[overlap])
        
                for index in range(0,len(self.indices_list),2):
                    for order in orders:
                        if btengan == chunks[chunk]:
                            btengan = 0
                            self.Error_map[chunk,overlap,order-3] = summ/chunks[chunk]
                        else:
                            summ += self.piece_interpolation(self.x_axis[self.indices_list[index]:self.indices_list[index+1]],
                                                  self.y_axis[self.indices_list[index]:self.indices_list[index+1]],order)
                            btengan += 1
                        progressbar_index += 1
                        self.Error_map_progressBar.setValue(int((progressbar_index/total_percentage)*100))
                        
        plt.imshow(self.Error_map[1,:,:], aspect='auto',
                   extent =[orders.min(), orders.max(), overlaps.min(), overlaps.max()],
                   cmap='PuRd', interpolation= 'bessel')
        plt.show()
        
    def draw_errormap(self):
        self.error_map()
        self.errorGraph.clear()
       
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.mkQApp()
        spec_plot = self.errorGraph.addPlot()
        img = pg.ImageItem()
        spec_plot.addItem(img)
        # hist = pg.HistogramLUTItem() # histogram to control the gradient of the image
        # hist.setImageItem(img)
        # graph.addItem(hist)

        # hist.setLevels(np.min(Sxx), np.max(Sxx))
        img.setImage(self.Error_map[1,:,:]) # Sxx: amplitude for each pixel
        # img.scale(t[-1]/np.size(self.Error_map[1,:,:], axis=1),
        #           f[-1]/np.size(self.Error_map[1,:,:], axis=0))
        
        spec_plot.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])
        spec_plot.setLabel('bottom', "Time", units='s')
        spec_plot.setLabel('left', "Frequency", units='Hz')
        # hist.gradient.restoreState({'ticks': [(0.0, (0, 0, 0, 255)), (0.01, (32, 0, 129, 255)),
        #                                     (0.8, (255, 255, 0, 255)), (0.5, (115, 15, 255, 255)),
        #                                     (1.0, (255, 255, 255, 255))], 'mode': 'rgb'})
        #(32, 0, 129, 255)
    
    def sliders_values(self):
        self.degree = self.order_horizontalSlider.value()
        self.overlap = self.overlap_horizontalSlider.value()
        self.chunks = self.chunks_horizontalSlider.value()
        
    def draw_signal(self):
        # self.pg.PlotCurveItem(self.x_axis, self.y_axis, pen = (1, 3), name= "Original")
        self.graphicsView.clear()
        self.graphicsView.plot(self.x_axis, self.y_axis, pen = self.pen, name= "Original", title="Orginal")
        self.sliders_values()
        self.devide_to_chunks(len(self.y_axis),self.chunks,self.overlap)
        # self.interpolation(self.x_axis, self.y_axis ,self.degree)
        self.draw_interpolation()
        self.mathTex_to_QPixmap()
        if self.actionSamples.isChecked():
            self.show_samples()

    def draw_interpolation(self):
        # x_axis = np.linspace(0, len(self.x_axis),len(self.x_axis))
        x_axis = np.linspace(0, self.x_axis[len(self.x_axis)-1], len(self.x_axis))

        y_int = np.zeros(len(self.y_axis))
        i = 1
        for chunk in range(0,self.chunks,1):
            self.interpolation([chunk+i-1,chunk+i],self.degree)
            y_int[self.index1:self.index2] += self.y_fitted
            i += 1
        # m = 0
        # for h in range(1,chunks,1):
        #     y_int[self.indices_list[2*h]:self.indices_list[2*h-1]+1] /= 2
        #     print(self.indices_list[2*h],self.indices_list[2*h-1])
        self.graphicsView.plot(x_axis, y_int, pen = 'g', name= "fitted")
    
    def error_button_text(self):
        if not self.flag:
            self.starterrorButton.setText('Cancel')
            self.flag = 1
        else:
            self.starterrorButton.setText('Start')
            self.flag = 0
            
    def portion(self,data_lenght,portion_percentage):
        self.indices_list =[0]
        last_index = int(((portion_percentage/100)*data_lenght)-1)
        self.indices_list.append(last_index)

    #_____________________________FETCHING DATA________________________#
    
    def prepare_data(self):
        self.browse()
        self.read_data()
        self.y_axis = self.amplitude
        self.x_axis = self.timestamps
        # self.x_axis = np.linspace(0, len(self.x_axis),len(self.x_axis))

        
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
        self.length_original_data = len(self.amplitude)
        # xy_axes = {'xaxis': xaxis_timestamps, 'yaxis': yaxis_values}
        # return xy_axes;

    
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
'''
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

    def interpolation(self,indices_list,order):
        self.length_original_data = len(self.signal)
        print(indices_list, 'indices', self.indices_list, 'selfindie')
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
        # if step > 3: 
        #     return 0
            
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
        print(b, 'bbbbb')
        for m in range(2,n+1):
            for i in range(0,n-m): 
                b[i,m] = (b[i+1,m-1]-b[i,m-1])/(x_samples[i+m-1]-x_samples[i])
                
        self.xx =np.arange(self.x[0],len(self.x),.5)
        np.around(self.xx, 2, self.xx)
        # print(self.xx[3],'hatoly el kalb dah')
        xt = 1
        # yint = np.zeros(len(self.xx))
        yint = b[1,1]
        ystring =str(b[1,1])
        xt_str =""
        coeff = b[1]
        np.around(coeff, 2, coeff)
        for m in range( 0,n-1):
            xt = xt*(self.xx-x_samples[m]) 
            yint = yint+b[1,m+1]*xt
            print(type(xt), 'ana xt')
            # if order > 4 and m > 2 and m < n-3:
            #     ystring += "."
            # else:
            xt_str += f"(x-{x_samples[m]})"
            new_coeff = coeff[m+1]
            if new_coeff < 0.1 and new_coeff > - 0.1 :continue
            ystring += f"+({new_coeff})({xt_str})"
        self.ystring = ystring
        print( type(yint), 'two')
        print(b, 'bbbbbeeeb')
        if step == 2:
            plt.plot(self.xx,yint)
            plt.show()
        self.xx = list(self.xx)
        print(type(yint), 'three')
        self.yint = yint
        self.gamal = self.xx
        yint_picked = [] 
        print(type(yint), 'four')
        for z in x:
            # print(yint, 'hoa yint')
            index = self.xx.index(z)
            yint_picked.append(self.yint[index])
        result = map(lambda original_data, interpolated_data: abs((interpolated_data - original_data)/original_data), y, yint_picked)
        result = list(result)
        error = sum(result[1:-1])/len(result[1:-1])
        return error '''