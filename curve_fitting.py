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
        self.error = 0
        self.formulae = []
        self.pen=pg.mkPen(color='c')
        # self.x=[2.3,3.5,4.6,5.0,6.1,7.0,7.2,8.5,8.6,8.8,8.9, 9, 9.1, 9.6, 10., 10.5]
        # self.signal=[1.2,2.6,3.1,4.2,5.3,5.8,6.4,7.0,7.6,8.0,8.5, 8.7, 9, 9.5, 10.1, 10.6]
        
        
        self.actionSamples.triggered.connect(self.show_samples)
        self.actionPlot.triggered.connect(self.draw_signal)
        self.actionOpen.triggered.connect(self.prepare_data)
        self.actionstring.triggered.connect(self.controlling_error_axes)
        self.starterrorButton.pressed.connect(self.controlling_error_axes)
        self.order_horizontalSlider.valueChanged.connect(self.run_sliders)
        self.chunks_horizontalSlider.valueChanged.connect(self.run_sliders)
        self.overlap_horizontalSlider.valueChanged.connect(self.run_sliders)
        self.third_dimensionSlider.valueChanged.connect(self.controlling_error_axes)
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
            for i in reversed(range(self.formulaeLayout.count())): 
                self.formulaeLayout.itemAt(i).widget().setParent(None)
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
        n = len(self.coeff)
        terms = [self.coeff[n-1]]
        terms = str(terms)
        for i in range(1,self.degree+1,1):
            if self.coeff[n-i] < 0.001 and self.coeff[n-i] > - 0.001 :continue
            # print(i, 'i', self.coeff[n-i])
            # coeff = self.coeff[i]
            terms += f'$+ ({self.coeff[n-i]}) x^{i} $'
            # terms.append(expr)
                
        # print(terms)
        mathTex = 'f(x)=' + terms
        # print(mathTex)
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
            # _thread.start_new_thread(self.draw_errormap,())
            self.draw_errormap()
            # t.start()
            self.start_error = 0
            # print(self.start_error)
        else:
            # _thread.exit()
            self.start_error = 1
        self.error_button_text()
        
    def controlling_error_axes(self):
        # self.Error_map = np.zeros((len(chunks),len(overlaps),len(orders)))
        if not self.error:
            self.error_map()
            self.error = 1
        self.third_dimension = self.third_dimensionSlider.value()
        condition1 = self.order_x_radioButton.isChecked() and self.order_y_radioButton.isChecked()
        condition2 = self.chunks_x_radioButton.isChecked() and self.chunks_y_radioButton.isChecked()
        condition3 = self.overlap_x_radioButton.isChecked() and self.overlap_y_radioButton.isChecked()
        if (condition1) or (condition2) or (condition3):
            self.label_alert.setText("Pick different variables for the same graph!😑")
        else:
            self.label_alert.setText("")
        self.third_dimensionSlider.setMinimum(1)
        if self.order_x_radioButton.isChecked() and self.chunks_y_radioButton.isChecked():
            self.img_map=self.Error_map[:,int(self.third_dimension),:]
            self.third_dimensionSlider.setMaximum(5)
        elif self.order_x_radioButton.isChecked() and self.overlap_y_radioButton.isChecked():
            self.img_map=self.Error_map[int(self.third_dimension),:,:]
            self.third_dimensionSlider.setMaximum(49)
        elif self.chunks_x_radioButton.isChecked() and self.order_y_radioButton.isChecked():
            self.img_map=np.transpose(self.Error_map[:,int(self.third_dimension),:])
            self.third_dimensionSlider.setMaximum(5)
        elif self.chunks_x_radioButton.isChecked() and self.overlap_y_radioButton.isChecked():
            self.img_map=self.Error_map[:,:,int(self.third_dimension)]
            self.third_dimensionSlider.setMaximum(20)
        elif self.overlap_x_radioButton.isChecked() and self.order_y_radioButton.isChecked():
            self.img_map=np.transpose(self.Error_map[int(self.third_dimension),:,:])
            self.third_dimensionSlider.setMaximum(49)
        elif self.overlap_x_radioButton.isChecked() and self.chunks_y_radioButton.isChecked():
            self.img_map=np.transpose(self.Error_map[:,:,int(self.third_dimension)])
            self.third_dimensionSlider.setMaximum(20)
        self.draw_errormap(self.img_map)
        
            
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
        # print(indices_list, 'indices', self.indices_list, 'selfindie')
        first_ind = indices_list[0]
        self.index1 = self.indices_list[first_ind]
        self.index2 = self.indices_list[indices_list[1]]+1
        y=np.array(self.y_axis[self.index1:self.index2])
        x=np.array(self.x_axis[self.indices_list[indices_list[0]]:self.indices_list[indices_list[1]]+1])
        self.piece_interpolation(x, y, degree)
        

    def piece_interpolation(self, x_data, y_data, degree):
        
        self.coeff = np.polyfit(x_data, y_data,degree, rcond=None, full=False, w=None, cov=False)
        self.y_fitted = np.polyval(self.coeff, x_data)
        # print(self.coeff)
        
        
        error = (self.y_fitted - y_data)/y_data
        # print((self.y_fitted - y_data) , 'fitted')
        error = np.abs(error)
        # print(len(error), 'len')
        error = sum(error)/len(error)
        # print(error, 'len')
        return error
    
    def error_map(self):
        chunks= np.arange(1,21)
        overlaps = np.arange(0,25, 6)
        orders = np.arange(1,50)
        progressbar_index = 0
        progressbar_percentage = 0
        total_percentage = (len(chunks))*(len(overlaps))*(len(orders))
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
                            self.Error_map[chunk,overlap,order-1] = summ/chunks[chunk]
                        else:
                            summ += self.piece_interpolation(self.x_axis[self.indices_list[index]:self.indices_list[index+1]],
                                                  self.y_axis[self.indices_list[index]:self.indices_list[index+1]],order)
                            btengan += 1
                        progressbar_index += 1
                        self.Error_map_progressBar.setValue(int((progressbar_index/total_percentage)*100))
                        
        plt.imshow(self.Error_map[:,3,:], aspect='auto',
                   extent =[orders.min(), orders.max(), overlaps.min(), overlaps.max()],
                   cmap='PuRd', interpolation= 'bessel')
        plt.show()
        print(self.Error_map[:,3,:])
        # plt.savefig('pic.png')
        
    def draw_errormap(self, error_map_slice):
        self.errorGraph.clear()
       
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.mkQApp()
        spec_plot = self.errorGraph.addPlot()
        img = pg.ImageItem()
        spec_plot.addItem(img)

        img.setImage(error_map_slice, cmap='PuRd') # Sxx: amplitude for each pixel

    
    def sliders_values(self):
        self.degree = self.order_horizontalSlider.value()
        self.overlap = self.overlap_horizontalSlider.value()
        self.chunks = self.chunks_horizontalSlider.value()
        self.third_dimension = self.third_dimensionSlider.value()
        
    def draw_signal(self):
        # self.pg.PlotCurveItem(self.x_axis, self.y_axis, pen = (1, 3), name= "Original")
        self.graphicsView.clear()
        self.graphicsView.plot(self.x_axis, self.y_axis, pen = self.pen, name= "Original", title="Orginal")
        self.sliders_values()
        self.devide_to_chunks(len(self.y_axis),self.chunks,self.overlap)
        # self.interpolation(self.x_axis, self.y_axis ,self.degree)
        self.draw_interpolation()
        # self.mathTex_to_QPixmap()
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
            self.mathTex_to_QPixmap()
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
        self.y_axis = np.array(self.amplitude)
        self.x_axis = np.array(self.timestamps)
        print(type(self.y_axis))
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
