import os
import sys
import nibabel
import numpy as np
import imageio
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QDialog, QFileDialog, QStackedWidget
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.uic import loadUi
import cv2

class Nifti(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('nifti.ui',self)

        self.CargarImagenes.clicked.connect(self.Cargar)
        self.Browse.clicked.connect(self.Buscar)
        #self.CambiarCorte.clicked.connect(self.Cargar)
        self.Corte.returnPressed.connect(self.Cargar)
        self.Avanzar.clicked.connect(self.CambiarPestana) 
        #Aca no se como hacer para mandarlo a la otra pestaña



    def CambiarPestana(self):
        pestaña_recorte.exec()
        widget.setCurrentIndex(1)

    def Buscar(self):
        ruta = os.getcwd()
        filename=QFileDialog.getOpenFileName(self,'Open file',ruta)
        self.RutaArchivo.setText(filename[0]) 

    def Cargar(self):
        #global ruta
        try:
            ruta_nii = self.RutaArchivo.text()
            img = nibabel.load(ruta_nii).get_fdata()
            
            corte = int(self.Corte.text())
            global Imagen_actual
            Imagen_actual = np.flip(np.transpose(img[:,corte,:]))
                
            alfa = 255/np.max(Imagen_actual)  #Para convertir a uint8
            Imagen_actual = cv2.convertScaleAbs(Imagen_actual, alpha = alfa)    

            global ruta_imagen
            ruta_imagen = 'graficar.png'
            imageio.imwrite(ruta_imagen,Imagen_actual)
            qpixmap = QPixmap(ruta_imagen)
            
            self.Imagen.setPixmap(qpixmap)
            self.Imagen.setScaledContents(True)
            max = str(img.shape[1])
            self.NivelMax.setText(max)
            self.error.setText('')
        except:
            if self.RutaArchivo.text() == '':
                self.error.setText('Ningún archivo seleccionado')
            else:
                self.error.setText('No se pudo cargar el archivo')

class Hipocampo(QDialog):
     
    def __init__(self):
        super().__init__()
        loadUi('gui_mia.ui',self)

        self.window_width, self.window_height = 600, 400        
        self.setWindowTitle('Alzheimer')
        self.setMinimumSize(self.window_width,self.window_height)

        self.CarguéImagen = False       #Si no se cargó imagen, no va a permitir dibujar rectangulos
        self.YaRecortéDos = False       #Si ya hizo dos recortes, no va a permitir seguir dibujando rectangulos
        
        self.pix = QPixmap(self.rect().size())
        #self.pix.fill(Qt.transparent)
        self.pix.fill(Qt.red)

                
        self.Recorte1 = []
        self.Recorte2 = []

        self.coordenadas_recorte1 = [0,0,0,0]   #[x_inicio, y_inicio, x_final, y_final]
        self.coordenadas_recorte2 = [0,0,0,0]   #[x_inicio, y_inicio, x_final, y_final]
        

 
        #self.pix = QPixmap('img.png').scaled(250,350,aspectRatioMode=Qt.IgnoreAspectRatio)


        self.begin, self.destination = QPoint() , QPoint()
        self.Click = False

        self.MostrarValores.clicked.connect(self.Results)
        self.ResetSeleccion.clicked.connect(self.ResetearSeleccion)
        self.Volver.clicked.connect(self.VolverInicio)

        self.path_imagen = ''

        layout = QVBoxLayout()
        self.setLayout(layout)

    def exec(self):
   
        self.pix.fill(Qt.transparent)


        current_directory = os.getcwd()
        
        if ruta_imagen == '':
            self.CarguéImagen = False
        else:
            self.CarguéImagen = True

        self.path_imagen = current_directory + '\\' + ruta_imagen
        self.path_imagen = self.path_imagen.replace('\\','/')
        qpixmap = QPixmap(self.path_imagen).scaled(width,height,aspectRatioMode = Qt.IgnoreAspectRatio)
        self.pix = qpixmap
        self.update()

    def paintEvent(self,event):
 
        if self.CarguéImagen:# and (not self.YaRecortéDos):
            
            painter = QPainter(self)
            painter.drawPixmap(QPoint(),self.pix)

            if not self.begin.isNull() and not self.destination.isNull() and (not self.YaRecortéDos):
                rect = QRect(self.begin,self.destination)
                painter.setPen(Qt.darkRed)
                painter.drawRect(rect.normalized())
            else:
                self.begin,self.destination = QPoint() , QPoint()
        else:
            self.begin,self.destination = QPoint() , QPoint()
        self.update()

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            posx = event.pos().x()
            posy = event.pos().y()
            if posx < width and posy < height:
                self.begin = event.pos()
                self.destination = self.begin
                self.Click = True
                self.update() 

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton & self.Click:
            self.destination = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):

        posx = event.pos().x()
        posy = event.pos().y()
        if event.button() & Qt.LeftButton & self.CarguéImagen & (not self.YaRecortéDos) & (posx < width) & (posy < height) & self.Click:
            self.Click = False
            rect = QRect(self.begin,self.destination)
            painter = QPainter(self.pix)
            painter.setPen(Qt.red)
            painter.drawRect(rect.normalized())


            factor_compresion_Y = len(Imagen_actual)/height
            factor_compresion_X = len(Imagen_actual[0])/width

            if (self.coordenadas_recorte1 == [0,0,0,0]) and (self.coordenadas_recorte2 == [0,0,0,0]):


                a = int(factor_compresion_X*self.begin.x())   
                b = int(factor_compresion_Y*self.begin.y())
                c = int(factor_compresion_X*self.destination.x())
                d = int(factor_compresion_Y*self.destination.y())

                if a > c:  #Estos dos if son para poder dibujar el cuadrado no solo desde arriba a la izquierda hacia abajo a la derecha
                    aux = a
                    a = c
                    c = aux
                if b > d:
                    aux = b
                    b = d
                    d = aux                   
                
                self.coordenadas_recorte1 = [a,b,c,d]
                self.Recorte1 = Imagen_actual[self.coordenadas_recorte1[1] : self.coordenadas_recorte1[3], self.coordenadas_recorte1[0] : self.coordenadas_recorte1[2]]            

                imageio.imwrite('graficar.png',self.Recorte1)
                qpixmap = QPixmap('graficar.png')
                self.LabelRecorte1.setPixmap(qpixmap)
                self.LabelRecorte1.setScaledContents(True)
                imageio.imwrite('graficar.png',Imagen_actual)

            elif self.coordenadas_recorte2 == [0,0,0,0]:
                a = int(factor_compresion_X*self.begin.x())      
                b = int(factor_compresion_Y*self.begin.y())
                c = int(factor_compresion_X*self.destination.x())
                d = int(factor_compresion_Y*self.destination.y())
                
                if a > c:  #Estos dos if son para poder dibujar el cuadrado no solo desde arriba a la izquierda hacia abajo a la derecha
                    aux = a
                    a = c
                    c = aux
                if b > d:
                    aux = b
                    b = d
                    d = aux  

                self.coordenadas_recorte2 = [a,b,c,d]
                self.Recorte2 = Imagen_actual[self.coordenadas_recorte2[1] : self.coordenadas_recorte2[3], self.coordenadas_recorte2[0] : self.coordenadas_recorte2[2]]            

                imageio.imwrite('graficar.png',self.Recorte2)
                qpixmap = QPixmap('graficar.png')
                self.LabelRecorte2.setPixmap(qpixmap)
                self.LabelRecorte2.setScaledContents(True)
                imageio.imwrite('graficar.png',Imagen_actual)
           
                self.YaRecortéDos = True

            self.begin, self.destination = QPoint() , QPoint()
            self.update()

    def ResetearSeleccion(self):
        self.YaRecortéDos = False
        self.coordenadas_recorte1 = [0,0,0,0]
        self.coordenadas_recorte2 = [0,0,0,0]
        
        pixmap = QPixmap(self.rect().size())
        pixmap.fill(Qt.white)
        self.LabelRecorte1.setPixmap(pixmap)
        self.LabelRecorte2.setPixmap(pixmap)
        
        self.pix.fill(Qt.transparent)

        qpixmap = QPixmap(self.path_imagen).scaled(width,height,aspectRatioMode = Qt.IgnoreAspectRatio)
        self.pix = qpixmap
        self.update()


    def VolverInicio(self):
        widget.setCurrentIndex(0)

   
   
    def Results(self):
        a = 1
        b = 2
        result = str(a+b)
        self.TextoResults.setText(result)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(''' 
    QWidget{
        font-size = 30px;

    }
    ''')



    Imagen_actual = []
    ruta_imagen = ''
    width = 250     #Ancho de la imagen donde se hacen los recortes
    height = 350    #Alto de la imagen donde se hacen los recortes

    widget = QStackedWidget()
    pestaña_principal = Nifti()
    pestaña_recorte = Hipocampo()
    widget.addWidget(pestaña_principal)  
    widget.addWidget(pestaña_recorte)
    widget.setCurrentIndex(0)
    widget.setFixedHeight(515)
    widget.setFixedWidth(723)
    widget.show()  


    #myApp = MyApp()
    #myApp.show()

    try:
        sys.exit(app.exec_())
    except:
        print('Saliendo del programa')
