import os
import sys
import cv2
import math
from numpy.lib.index_tricks import MGridClass
import nibabel
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler  
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, QDialog, QFileDialog, QStackedWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.uic import loadUi

###################################### FUNCIONES DE TEXTURAS #########################################################
def GLCM(img, dir):

  while (dir != 0 and dir !=45 and dir != 90 and dir != 135):
    print('La dirección pedida no se corresponde con ninguna de las direcciones posibles.')
    dir = int(input('Ingrese una opción válida (0/45/90/135) :'))

  filas = img.shape[0]
  cols = img.shape[1]
  
  max = np.max(img)
  min = np.min(img)

  if max-min == 0:
    matriz = np.ones((1,1))
  else:
    #creo la matriz teniendo en cuenta los valores mínimo y máximo de gris de la imagen
    matriz = np.zeros((max-min+1,max-min+1))

    if dir==0:
      #Dirección horizontal
      for i in range(filas):
        for j in range(cols-1):
          a = img[i,j]
          b = img[i,j+1]
          indice_a = a-min
          indice_b = b-min
          matriz[indice_a,indice_b]+=1
          matriz[indice_b,indice_a]+=1
          #sumo en la posición de la transición y en la posición simétrica
          #si llega a ser de la diagonal se suma dos veces

    if dir==45:
      #Dirección diagonal para abajo
      for i in range(1,filas):
        for j in range(cols-1):
          a = img[i,j]
          b = img[i-1,j+1]
          indice_a = a-min
          indice_b = b-min
          matriz[indice_a,indice_b]+=1
          matriz[indice_b,indice_a]+=1
          #sumo en la posición de la transición y en la posición simétrica
          #si llega a ser de la diagonal se suma dos veces

    if dir==90:
      #Dirección vertical
      for i in range(filas-1):
        for j in range(cols):
          a = img[i,j]
          b = img[i+1,j]
          indice_a = a-min
          indice_b = b-min
          matriz[indice_a,indice_b]+=1
          matriz[indice_b,indice_a]+=1
          #sumo en la posición de la transición y en la posición simétrica
          #si llega a ser de la diagonal se suma dos veces

    if dir==135:
      #Dirección diagonal para arriba
      for i in range(filas-1):
        for j in range(cols-1):
          a = img[i,j]
          b = img[i+1,j+1]
          indice_a = a-min
          indice_b = b-min
          matriz[indice_a,indice_b]+=1
          matriz[indice_b,indice_a]+=1
          #sumo en la posición de la transición y en la posición simétrica
          #si llega a ser de la diagonal se suma dos veces


    #Normalizo la matriz
    factor =  2*((max-min+1)**2)
    matriz = matriz/factor
  
  return matriz, min 
  #Devuelvo la GLCM en la dirección pedida y el mínimo nivel de gris encontrado

def Media(matriz, min):
  filas = matriz.shape[0]
  cols = matriz.shape[1]
  
  media = 0
  for i in range(filas):
    for j in range(cols):
      media += (i+min)*matriz[i,j]

  return media

def Desvio(matriz, media, min):
  filas = matriz.shape[0]
  cols = matriz.shape[1]

  desvio = 0
  for i in range(filas):
    for j in range(cols):
      desvio += ((i+min-media)**2)*matriz[i,j]  

  return desvio

def Entropia(matriz):
  filas = matriz.shape[0]
  cols = matriz.shape[1]

  entropia = 0
  for i in range(filas):
    for j in range(cols):
      if (matriz[i,j] != 0):
        entropia += matriz[i,j]*np.log2(matriz[i,j])

  return entropia

def Energia(matriz):
  filas = matriz.shape[0]
  cols = matriz.shape[1]

  energia = 0
  for i in range(filas):
    for j in range(cols):
      energia += matriz[i,j]**2
  
  return energia

def LH(matriz):
  filas = matriz.shape[0]
  cols = matriz.shape[1]

  lh = 0
  for i in range(filas):
    for j in range(cols):
      lh += (matriz[i,j]**2)/(1+(i-j)**2) ##Chequear, creo que así tiene más sentido

  return lh

def Contraste(matriz):
  filas = matriz.shape[0]
  cols = matriz.shape[1]

  contraste = 0
  for i in range(filas):
    for j in range(cols):
      contraste += ((i-j)**2)*matriz[i,j]

  return contraste

def Correlacion(matriz,media,desvio,min):


  filas = matriz.shape[0]
  cols = matriz.shape[1]

  correlacion = 0
  for i in range(filas):
    for j in range(cols):
      correlacion += ((i+min-media)*(j+min-media)*matriz[i,j])/(desvio**2)
      
  return correlacion

def Metricas(img,angulo):  
  M = GLCM(img, angulo)
  
  lista = []

  media = Media(M[0],M[1])
  lista.append(np.round(media,2))
  desvio = Desvio(M[0],media,M[1])
  lista.append(np.round(desvio,2))
  entropia = Entropia(M[0])
  lista.append(np.round(entropia,2))
  energia = Energia(M[0])
  lista.append(np.round(energia,10))
  lh = LH(M[0])
  lista.append(np.round(lh,10))
  contraste = Contraste(M[0])
  lista.append(np.round(contraste,2))
  corr = Correlacion(M[0],media,desvio,M[1])
  lista.append(np.round(corr,4))

  return lista

def ArmarProm(izq,der):
    lista = []
    for i in range(len(izq)):
        aux = (izq[i]+der[i])/2
        lista.append(aux)
    return lista

def redondear(numero, digitos):
    return round(numero, digitos - int(math.floor(math.log10(abs(numero)))) - 1)

#####################################################################################################################


class Inicio(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('gui_inicio.ui',self)

        self.window_width, self.window_height = 600, 400        
        self.setWindowTitle('Inicio')
        self.setMinimumSize(self.window_width,self.window_height)
        self.Avanzar.clicked.connect(self.Adelante)
        self.MostrarMatriz.clicked.connect(self.Graficar)
        self.EntrenarRed.clicked.connect(self.Entrenar)

        qpixmap = QPixmap('logo_sinusense.png')
        self.logo.setPixmap(qpixmap)
        self.logo.setScaledContents(True)

        # Para matriz de confusion

    def Adelante(self):
        widget.setCurrentIndex(1)

    def Graficar(self):
        class_names = ['CN', 'AD']
        plt.close()
        disp = plot_confusion_matrix(clf_t, X_test, y_test, display_labels = class_names, cmap = plt.cm.Blues)
        disp.ax_.set_title('Matriz de confusión')
        plt.savefig('confusion.png')
        qpixmap = QPixmap('confusion.png')
        self.Imagen.setPixmap(qpixmap)
        self.Imagen.setScaledContents(True)    
        os.remove('confusion.png')
        # self.Imagen
        # NO SE COMO PONER LA MATRIZ DE CONFUSION EN LA IMAGEN (self.Imagen)
        global CM
        global exac
        CM = disp.confusion_matrix
        a = CM[0,0]
        b = CM[0,1]
        c = CM[1,0]
        d = CM[1,1]
        exac = int((a+d)*100/(a+b+c+d))
        prec = int(d*100/(b+d))
        sens = int(d*100/(d+c))
        espe = int(a*100/(a+b))
        self.ResultadosLabel.setText('Métricas del clasificador')
        self.ExLabel.setText('Exactitud:')
        self.PreLabel.setText('Precisión:')
        self.SenLabel.setText('Sensibilidad:')
        self.EsLabel.setText('Especificidad:')
        self.Exactitud.setText(str(exac))
        self.Precision.setText(str(prec))
        self.Sensibilidad.setText(str(sens))
        self.Especificidad.setText(str(espe))
        self.Porc1.setText('%')
        self.Porc2.setText('%')
        self.Porc3.setText('%')
        self.Porc4.setText('%')

    def Entrenar(self):
        global X
        global y

        # Orden de las features de la matriz X:
        # 0 - Edad, 1 - Sexo, 2 - MMSE, 3 - Media, 4 - Desvío, 5 - Entropía, 6 - Energía, 7 - LH, 8 - Contraste, 9 - Correlación
        X = np.array(np.loadtxt('input.txt'))
        X = np.reshape(X,(80,10))
        global EntropiaBase
        EntropiaBase = abs(min(X[:,5]))
        X[:,5] = X[:,5] + EntropiaBase
        # X = np.array(np.loadtxt('input_sinentropia.txt'))
        # X = np.reshape(X,(80,9))
        y = np.append([1]*40,[2]*40)

        # Clasificador de prueba
        global X_test
        global y_test

        # scaler_t = StandardScaler()  
        # scaler_t.fit(X_train)  
        # X_train = scaler_t.transform(X_train)  
        # X_test = scaler_t.transform(X_test)  

        X_new = SelectKBest(chi2, k=8).fit_transform(X, y)
        # print(X[0])
        # print(X_new[0])
        # Al parecer el KBest descarta las métricas energía y LH
        # Nuevo orden de las features de la matriz X:
        # 0 - Edad, 1 - Sexo, 2 - MMSE, 3 - Media, 4 - Desvío, 5 - Entropía, 6 - Contraste, 7 - Correlación
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 0)

        global clf_t
        # clf_t = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state = 1)      # Red neuronal
        clf_t = ensemble.RandomForestClassifier()
        clf_t.fit(X_train, y_train)

        # Clasificador real
        # scaler = StandardScaler()  
        # scaler.fit(X)  
        # X = scaler.transform(X)  
        global clf
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state = 1)      # Red neuronal
        clf = ensemble.RandomForestClassifier()
        clf.fit(X_new, y)
        self.Entrenando.setText('El clasificador ha sido entrenado exitosamente')


class Nifti(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('nifti.ui',self)

        self.CargarImagenes.clicked.connect(self.Cargar)
        self.Browse.clicked.connect(self.Buscar)
        #self.CambiarCorte.clicked.connect(self.Cargar)
        self.Corte.returnPressed.connect(self.Cargar)
        self.Avanzar.clicked.connect(self.Adelante)
        self.Volver.clicked.connect(self.Atras)

        qpixmap = QPixmap('logo_sinusense.png')
        self.logo.setPixmap(qpixmap)
        self.logo.setScaledContents(True)    

    def Atras(self):
        widget.setCurrentIndex(0)

    def Adelante(self):
        pestaña_recorte.exec()
        widget.setCurrentIndex(2)

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
            max = str(img.shape[1]-1)
            self.NivelMax.setText(max)
            self.error.setText('')
        except:
            if self.RutaArchivo.text() == '':
                msg=QMessageBox()
                msg.setWindowTitle('Error')
                msg.setText('Ningún archivo seleccionado. Seleccione alguna imagen.')
                msg.setIcon(QMessageBox.Critical)
                msg.exec()
                #self.error.setText('Ningún archivo seleccionado')
            else:
                msg=QMessageBox()
                msg.setWindowTitle('Error')
                msg.setText('No se pudo cargar el archivo. Por favor ingrese un archivo NIFTI (.nii) y revise el límite de capas.')
                msg.setIcon(QMessageBox.Warning)
                msg.exec()
                #self.error.setText('No se pudo cargar el archivo')

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
    

        self.begin, self.destination = QPoint() , QPoint()
        self.Click = False

        qpixmap = QPixmap('logo_sinusense.png')
        self.logo.setPixmap(qpixmap)
        self.logo.setScaledContents(True)

        self.ResetSeleccion.clicked.connect(self.ResetearSeleccion)
        self.Volver.clicked.connect(self.Atras)
        self.Avanzar.clicked.connect(self.Adelante)
        self.CargarDatos.clicked.connect(self.Cargar)
        self.Ayuda.clicked.connect(self.MensajeAyuda)
        self.AyudaRecortes.clicked.connect(self.MensajeAyudaRecortes)

        self.path_imagen = ''

        layout = QVBoxLayout()
        self.setLayout(layout)

    
    def MensajeAyuda(self):
        msg = QMessageBox()
        msg.setWindowTitle('Ayuda')
        msg.setText('Seleccione primero el hipocampo de un hemisferio y luego el del otro. Si no está conforme con el resultado, puede realizarlo nuevamente con el botón "Resetear selección". Una vez que tenga los recortes, complete los campos solicitados, cargue los datos, y avance a la siguiente pantalla.')
        msg.setIcon(QMessageBox.Information)    
        msg.exec()

    def MensajeAyudaRecortes(self):
        msg = QMessageBox()
        msg.setWindowTitle('¿Cómo recortar?')
        msg.setText('Haga click y mantenga presionado para dibujar un recorte. Intente hacer el recorte del hipocampo lo más ajustado posible, sin perder segmentos del mismo.')
        msg.setIcon(QMessageBox.Information)    
        msg.exec()

    def Cargar(self):
        global edad
        global sexo
        global puntajeMiniMental
        self.error.setText('')
        
        sexo = self.Sex.text()
        if sexo not in ['F', 'M']:
            msg = QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText('El sexo debe ingresarse como F (femenino) o M (masculino).')
            msg.setIcon(QMessageBox.Warning)    
            msg.exec()
            #self.error.setText('El sexo debe ingresarse como F (femenino) o M (masculino)')
        
            str1 = 'color:red; background-color: transparent;font: 75 10pt "Helvetica"'
            self.error.setStyleSheet(str1) 
        try:
            edad = int(self.Age.text())
            puntajeMiniMental = int(self.MiniMentalScore.text())
            if edad < 0:
                msg = QMessageBox()
                msg.setWindowTitle('Error')
                msg.setText('La edad debe ser un valor positivo.')
                msg.setIcon(QMessageBox.Warning)    
                msg.exec()
                #self.error.setText('La edad y puntaje mini-mental ingresados deben ser positivos')
                
                str1 = 'color:red; background-color: transparent;font: 75 10pt "Helvetica"'
                self.error.setStyleSheet(str1)            
            
            elif puntajeMiniMental < 0 or puntajeMiniMental >30:
                msg = QMessageBox()
                msg.setWindowTitle('Error')
                msg.setText('El puntaje mini-mental ingresado debe ser un entero y estar entre 0 y 30.')
                msg.setIcon(QMessageBox.Warning)    
                msg.exec()

            elif sexo in ['F','M']:
                self.error.setText('Los datos fueron cargados correctamente')
                
                str1 = 'color:black; background-color: transparent;font: 75 10pt "Helvetica"'
                self.error.setStyleSheet(str1)


        except:
            self.error.setText('La edad y puntaje mini-mental ingresados deben ser numéricos')
            str1 = 'color:red; background-color: transparent;font: 75 10pt "Helvetica"'
            self.error.setStyleSheet(str1)   

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
 
        if self.CarguéImagen:
            
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

            area_recorte = (c-a)*(d-b) 

            if (self.coordenadas_recorte1 == [0,0,0,0]) and (self.coordenadas_recorte2 == [0,0,0,0]) and area_recorte > 2:
       
                self.coordenadas_recorte1 = [a,b,c,d]
                self.Recorte1 = Imagen_actual[self.coordenadas_recorte1[1] : self.coordenadas_recorte1[3], self.coordenadas_recorte1[0] : self.coordenadas_recorte1[2]]            

                imageio.imwrite('graficar.png',self.Recorte1)
                qpixmap = QPixmap('graficar.png')
                self.LabelRecorte1.setPixmap(qpixmap)
                self.LabelRecorte1.setScaledContents(True)
                imageio.imwrite('graficar.png',Imagen_actual)

            elif self.coordenadas_recorte2 == [0,0,0,0] and area_recorte > 2:

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
        
        self.Recorte1 = []
        self.Recorte2 = []

        pixmap = QPixmap(self.rect().size())
        pixmap.fill(Qt.white)
        self.LabelRecorte1.setPixmap(pixmap)
        self.LabelRecorte2.setPixmap(pixmap)
        
        self.pix.fill(Qt.transparent)

        qpixmap = QPixmap(self.path_imagen).scaled(width,height,aspectRatioMode = Qt.IgnoreAspectRatio)
        self.pix = qpixmap
        self.update()


    def Atras(self):
        # pestaña_principal.exec()
        widget.setCurrentIndex(1)

    def Adelante(self):
        pestaña_resultados.Diagnostico()
        widget.setCurrentIndex(3)
   

class Resultados(QDialog):
     
    def __init__(self):
        super().__init__()
        loadUi('gui_res.ui',self)

        self.window_width, self.window_height = 600, 400        
        self.setWindowTitle('Resultados')
        self.setMinimumSize(self.window_width,self.window_height)
        
        self.Volver.clicked.connect(self.Atras)

        qpixmap = QPixmap('logo_sinusense.png')
        self.logo.setPixmap(qpixmap)
        self.logo.setScaledContents(True)

    def Atras(self):
        # pestaña_principal.exec()
        widget.setCurrentIndex(2)

    def ArmarInput(self):

        try:
            metricas_recorte1 = Metricas(np.array(pestaña_recorte.Recorte1),0)
            metricas_recorte2 = Metricas(np.array(pestaña_recorte.Recorte2),0)

            metricas = ArmarProm(metricas_recorte1, metricas_recorte2)

            self.Metrica1.setText(str(redondear(metricas[0],3)))
            self.Metrica2.setText(str(redondear(metricas[1],3)))
            self.Metrica3.setText(str(redondear(metricas[2],3)))
            self.Metrica4.setText(str(redondear(metricas[3],3)))
            self.Metrica5.setText(str(redondear(metricas[4],3)))
            self.Metrica6.setText(str(redondear(metricas[5],3)))
            self.Metrica7.setText(str(redondear(metricas[6],3)))
            self.errorRecortes.setText('')
        
        except:
            self.errorRecortes.setText('No se han seleccionado recortes')
            return [[False], [False]]  

        if edad == '' or puntajeMiniMental == '' or sexo == '' or edad < 0 or sexo not in ['F','M'] or puntajeMiniMental < 0:
            self.error.setText('Los datos de paciente no fueron cargados correctamente')
            return [[False], [False]]  
            
        else:
            x = np.zeros([1,8])
            x[0,0] = edad
            if sexo == 'F':
                x[0,1] = 1
            else:
                x[0,1] = 2
            x[0,2] = puntajeMiniMental
            x[0,3] = metricas[0]
            x[0,4] = metricas[1]
            x[0,5] = np.abs(metricas[2] + EntropiaBase)
            x[0,6] = metricas[5]
            x[0,7] = metricas[6]
            #print(x) 
            return x
    
    def Diagnostico(self):
        x = self.ArmarInput()
        self.Resultado.setText('')
        if clf == '':
            msg = QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText('El algoritmo no fue entrenado. Entrénelo en la pantalla de inicio.')
            msg.setIcon(QMessageBox.Warning)    
            msg.exec()
            #self.error.setText('El algoritmo no fue entrenado. Entrénelo en la pantalla de inicio.')

        elif any(x[0]) == False:    # Acá entra cuando no se cargaron correctamente los datos del paciente
            self.Resultado.setText('')
           
        else:
            self.error.setText('')
            resultado = clf.predict(x)

            if resultado == 1:
                self.Resultado.setText('No hay indicios de AD')
            if resultado == 2:
                self.Resultado.setText('Demencia AD')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(''' 
    QWidget{
        font-size = 30px;

    }
    ''')


    clf = '' #Para chequear en la pestaña de resultados si se modifica o no. Si sigue siendo '', es porque no se entrenó la red
    
    edad = ''    #Para chequear en la pestaña de resultados si se modifica o no. Si sigue siendo '', es porque no se cargó el dato
    puntajeMiniMental = ''  #Para chequear en la pestaña de resultados si se modifica o no. Si sigue siendo '', es porque no se cargó el dato
    sexo = '' #Para chequear en la pestaña de resultados si se modifica o no. Si sigue siendo '', es porque no se cargó el dato
    EntropiaBase = 0
    Imagen_actual = []
    ruta_imagen = ''
    width = 250     #Ancho de la imagen donde se hacen los recortes
    height = 350    #Alto de la imagen donde se hacen los recortes

    widget = QStackedWidget()
    pestaña_inicio = Inicio()
    pestaña_principal = Nifti()
    pestaña_recorte = Hipocampo()
    pestaña_resultados = Resultados()
    widget.addWidget(pestaña_inicio)
    widget.addWidget(pestaña_principal)  
    widget.addWidget(pestaña_recorte)
    widget.addWidget(pestaña_resultados)
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



