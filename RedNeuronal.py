import nibabel
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2


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

def Acondicionar(img,corte,filmin,filmax,colmin,colmax):

  img_mod = np.flip(np.transpose(img[:,corte,:]))
  alfa = 255/np.max(img_mod)
  img_mod = cv2.convertScaleAbs(img_mod, alpha = alfa)  #Para convertir a uint8
  img_final = img_mod[filmin:filmax,colmin:colmax]
  return img_final

def ArmarProm(izq,der):
    lista = []
    for i in range(len(izq)):
        aux = (izq[i]+der[i])/2
        lista.append(aux)
    return lista

def ArmarInput(prom,edad,sexo,mmse):
    lista = [edad,sexo,mmse]

    for elem in prom:
        lista.append(elem)
    
    return lista

###EMPIEZO A CARGAR IMAGENES###
S1 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/001_120.nii')).get_fdata()
S1_izq = Acondicionar(S1,120,134,155,53,75)
S1_der = Acondicionar(S1,120,134,155,100,122)
S2 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/005_113.nii')).get_fdata()
S2_izq = Acondicionar(S2,113,135,157,50,75)
S2_der = Acondicionar(S2,113,135,157,102,124)
S3 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/006_118.nii')).get_fdata()
S3_izq = Acondicionar(S3,118,137,157,50,75)
S3_der = Acondicionar(S3,118,137,157,102,125)
S4 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/007_120.nii')).get_fdata()
S4_izq = Acondicionar(S4,120,123,145,50,75)
S4_der = Acondicionar(S4,120,123,145,102,127)
S5 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/010_108.nii')).get_fdata()
S5_izq = Acondicionar(S5,108,123,145,55,77)
S5_der = Acondicionar(S5,108,127,145,96,115)
S6 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/014_115.nii')).get_fdata()
S6_izq = Acondicionar(S6,115,137,155,50,77)
S6_der = Acondicionar(S6,115,137,155,102,125)
S7 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/015_115.nii')).get_fdata()
S7_izq = Acondicionar(S7,115,140,157,52,75)
S7_der = Acondicionar(S7,115,139,157,104,127)
S8 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/018_120.nii')).get_fdata()
S8_izq = Acondicionar(S8,120,140,157,50,75)
S8_der = Acondicionar(S8,120,140,157,102,127)
S9 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/025_115.nii')).get_fdata()
S9_izq = Acondicionar(S9,115,135,157,52,75)
S9_der = Acondicionar(S9,115,137,157,102,125)
S10 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/033_115.nii')).get_fdata()
S10_izq = Acondicionar(S10,115,138,157,51,75)
S10_der = Acondicionar(S10,115,139,157,102,125)
'''
S11 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/036_118.nii')).get_fdata()
S11_izq = Acondicionar(S11,118,138,157,51,75)
S11_der = Acondicionar(S11,118,139,157,102,125)
S12 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/037_115.nii')).get_fdata()
S12_izq = Acondicionar(S12,115,138,157,53,77)
S12_der = Acondicionar(S12,115,137,157,102,127)
S13 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/046_115.nii')).get_fdata()
S13_izq = Acondicionar(S13,115,138,157,50,75)
S13_der = Acondicionar(S13,115,140,157,100,125)
S14 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/050_110.nii')).get_fdata()
S14_izq = Acondicionar(S14,110,122,145,57,75)
S14_der = Acondicionar(S14,110,120,145,98,117)
S15 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/060_112.nii')).get_fdata()
S15_izq = Acondicionar(S15,112,132,145,54,75)
S15_der = Acondicionar(S15,112,132,145,98,125)
S16 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/065_120.nii')).get_fdata()
S16_izq = Acondicionar(S16,120,136,152,50,75)
S16_der = Acondicionar(S16,120,136,152,100,125)
S17 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/077_115.nii')).get_fdata()
S17_izq = Acondicionar(S17,115,163,180,45,75)
S17_der = Acondicionar(S17,115,162,180,100,125)
S18 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/080_113.nii')).get_fdata()
S18_izq = Acondicionar(S18,113,135,152,52,75)
S18_der = Acondicionar(S18,113,137,150,100,126)
S19 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/090_120.nii')).get_fdata()
S19_izq = Acondicionar(S19,120,137,155,53,76)
S19_der = Acondicionar(S19,120,138,152,102,127)
S20 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/093_107.nii')).get_fdata()
S20_izq = Acondicionar(S20,107,127,145,55,80)
S20_der = Acondicionar(S20,107,125,143,100,120)
S21 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/099_117.nii')).get_fdata()
S21_izq = Acondicionar(S21,117,140,155,50,74)
S21_der = Acondicionar(S21,117,140,153,105,130)
S22 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/105_110.nii')).get_fdata()
S22_izq = Acondicionar(S22,110,135,155,55,80)
S22_der = Acondicionar(S22,110,135,150,103,120)
S23 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/113_110.nii')).get_fdata()
S23_izq = Acondicionar(S23,110,125,155,53,75)
S23_der = Acondicionar(S23,110,125,150,98,120)
S24 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/115_110.nii')).get_fdata()
S24_izq = Acondicionar(S24,110,135,155,53,77)
S24_der = Acondicionar(S24,110,135,155,100,123)
S25 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/122_112.nii')).get_fdata()
S25_izq = Acondicionar(S25,112,137,155,53,72)
S25_der = Acondicionar(S25,112,140,155,100,123)
S26 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/123_103.nii')).get_fdata()
S26_izq = Acondicionar(S26,103,117,140,53,74)
S26_der = Acondicionar(S26,103,120,140,100,120)
S27 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/125_115.nii')).get_fdata()
S27_izq = Acondicionar(S27,115,134,150,56,77)
S27_der = Acondicionar(S27,115,135,150,102,125)
S28 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/127_102.nii')).get_fdata()
S28_izq = Acondicionar(S28,102,117,140,50,75)
S28_der = Acondicionar(S28,102,115,135,100,122)
S29 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/154_112.nii')).get_fdata()
S29_izq = Acondicionar(S29,112,145,170,50,75)
S29_der = Acondicionar(S29,112,148,170,103,129)
S30 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/200_113.nii')).get_fdata()
S30_izq = Acondicionar(S30,113,130,150,50,75)
S30_der = Acondicionar(S30,113,136,155,100,125)
S31 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/204_113.nii')).get_fdata()
S31_izq = Acondicionar(S31,113,138,158,50,75)
S31_der = Acondicionar(S31,113,140,160,104,130)
S32 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/207_118.nii')).get_fdata()
S32_izq = Acondicionar(S32,118,138,158,53,76)
S32_der = Acondicionar(S32,118,140,160,102,125)
S33 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/211_115.nii')).get_fdata()
S33_izq = Acondicionar(S33,115,133,155,53,76)
S33_der = Acondicionar(S33,115,135,157,104,127)
S34 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/213_110.nii')).get_fdata()
S34_izq = Acondicionar(S34,110,120,140,54,75)
S34_der = Acondicionar(S34,110,120,140,97,121)
S35 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/219_113.nii')).get_fdata()
S35_izq = Acondicionar(S35,113,138,160,53,75)
S35_der = Acondicionar(S35,113,137,160,102,125)
S36 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/220_115.nii')).get_fdata()
S36_izq = Acondicionar(S36,115,138,160,51,75)
S36_der = Acondicionar(S36,115,140,160,102,125)
S37 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/221_115.nii')).get_fdata()
S37_izq = Acondicionar(S37,115,136,160,51,77)
S37_der = Acondicionar(S37,115,138,160,102,127)
S38 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/222_115.nii')).get_fdata()
S38_izq = Acondicionar(S38,115,133,160,49,74)
S38_der = Acondicionar(S38,115,135,160,102,127)
S39 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/227_113.nii')).get_fdata()
S39_izq = Acondicionar(S39,113,133,160,49,74)
S39_der = Acondicionar(S39,113,137,160,102,127)
S40 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Sano/229_113.nii')).get_fdata()
S40_izq = Acondicionar(S40,113,150,175,55,77)
S40_der = Acondicionar(S40,113,150,175,98,120)
'''

A1 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/024_105.nii')).get_fdata()
A1_izq = Acondicionar(A1,105,115,133,52,75)
A1_der = Acondicionar(A1,105,115,133,100,125)
A2 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/031_118.nii')).get_fdata()
A2_izq = Acondicionar(A2,118,137,155,50,75)
A2_der = Acondicionar(A2,118,142,160,100,127)
A3 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/076_115.nii')).get_fdata()
A3_izq = Acondicionar(A3,115,137,155,52,75)
A3_der = Acondicionar(A3,115,136,160,100,127)
A4 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/095_115.nii')).get_fdata()
A4_izq = Acondicionar(A4,115,133,155,52,75)
A4_der = Acondicionar(A4,115,133,157,103,127)
A5 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/100_109.nii')).get_fdata()
A5_izq = Acondicionar(A5,109,148,165,54,78)
A5_der = Acondicionar(A5,109,148,165,100,125)
A6 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/128_110.nii')).get_fdata()
A6_izq = Acondicionar(A6,110,125,145,57,79)
A6_der = Acondicionar(A6,110,127,147,100,125)
A7 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/136_110.nii')).get_fdata()
A7_izq = Acondicionar(A7,110,135,155,57,79)
A7_der = Acondicionar(A7,110,137,157,100,125)
A8 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/145_115.nii')).get_fdata()
A8_izq = Acondicionar(A8,115,140,160,50,77)
A8_der = Acondicionar(A8,115,141,162,102,128)
A9 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/150_115.nii')).get_fdata()
A9_izq = Acondicionar(A9,115,136,157,50,77)
A9_der = Acondicionar(A9,115,138,159,102,128)
A10 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/158_118.nii')).get_fdata()
A10_izq = Acondicionar(A10,118,136,157,50,77)
A10_der = Acondicionar(A10,118,138,159,102,128)
'''
A11 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/198_112.nii')).get_fdata()
A11_izq = Acondicionar(A11,112,140,160,42,70)
A11_der = Acondicionar(A11,112,140,162,102,132)
A12 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/199_112.nii')).get_fdata()
A12_izq = Acondicionar(A12,112,132,155,50,74)
A12_der = Acondicionar(A12,112,136,157,102,127)
A13 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/202_114.nii')).get_fdata()
A13_izq = Acondicionar(A13,114,137,155,50,74)
A13_der = Acondicionar(A13,112,138,157,102,129)
A14 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/206_115.nii')).get_fdata()
A14_izq = Acondicionar(A14,115,136,155,52,74)
A14_der = Acondicionar(A14,115,136,157,102,127)
A15 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/209_113.nii')).get_fdata()
A15_izq = Acondicionar(A15,113,140,155,48,74)
A15_der = Acondicionar(A15,113,140,157,102,128)
A16 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/212_113.nii')).get_fdata()
A16_izq = Acondicionar(A16,113,135,155,52,76)
A16_der = Acondicionar(A16,113,139,157,102,128)
A17 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/214_112.nii')).get_fdata()
A17_izq = Acondicionar(A17,112,137,155,48,76)
A17_der = Acondicionar(A17,112,139,157,102,132)
A18 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/217_109.nii')).get_fdata()
A18_izq = Acondicionar(A18,109,146,165,48,76)
A18_der = Acondicionar(A18,109,148,167,100,126)
A19 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/226_113.nii')).get_fdata()
A19_izq = Acondicionar(A19,113,139,165,45,74)
A19_der = Acondicionar(A19,113,138,167,104,131)
A20 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/238_116.nii')).get_fdata()
A20_izq = Acondicionar(A20,116,136,162,50,75)
A20_der = Acondicionar(A20,116,138,167,104,131)
A21 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/239_112.nii')).get_fdata()
A21_izq = Acondicionar(A21,112,136,160,48,75)
A21_der = Acondicionar(A21,112,138,164,105,130)
A22 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/244_120.nii')).get_fdata()
A22_izq = Acondicionar(A22,120,140,160,51,73)
A22_der = Acondicionar(A22,120,141,160,103,125)
A23 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/255_115.nii')).get_fdata()
A23_izq = Acondicionar(A23,115,135,157,51,73)
A23_der = Acondicionar(A23,115,135,160,103,130)
A24 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/262_117.nii')).get_fdata()
A24_izq = Acondicionar(A24,117,140,157,55,75)
A24_der = Acondicionar(A24,117,139,160,103,125)
A25 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/265_119.nii')).get_fdata()
A25_izq = Acondicionar(A25,119,140,157,52,75)
A25_der = Acondicionar(A25,119,141,160,103,128)
A26 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/267_115.nii')).get_fdata()
A26_izq = Acondicionar(A26,115,130,149,55,77)
A26_der = Acondicionar(A26,115,130,150,102,126)
A27 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/271_118.nii')).get_fdata()
A27_izq = Acondicionar(A27,118,138,154,52,77)
A27_der = Acondicionar(A27,118,138,155,102,129)
A28 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/281_108.nii')).get_fdata()
A28_izq = Acondicionar(A28,108,141,154,56,77)
A28_der = Acondicionar(A28,108,141,155,100,125)
A29 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/282_115.nii')).get_fdata()
A29_izq = Acondicionar(A29,115,136,154,51,77)
A29_der = Acondicionar(A29,115,137,155,100,128)
A30 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/286_113.nii')).get_fdata()
A30_izq = Acondicionar(A30,113,139,154,50,77)
A30_der = Acondicionar(A30,113,137,155,102,128)
A31 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/287_121.nii')).get_fdata()
A31_izq = Acondicionar(A31,121,139,160,47,77)
A31_der = Acondicionar(A31,121,142,160,102,129)
A32 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/292_118.nii')).get_fdata()
A32_izq = Acondicionar(A32,118,137,160,50,77)
A32_der = Acondicionar(A32,118,140,160,102,129)
A33 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/298_112.nii')).get_fdata()
A33_izq = Acondicionar(A33,112,136,157,50,77)
A33_der = Acondicionar(A33,112,138,160,102,127)
A34 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/312_118.nii')).get_fdata()
A34_izq = Acondicionar(A34,118,120,140,60,80)
A34_der = Acondicionar(A34,118,120,138,102,119)
A35 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/313_118.nii')).get_fdata()
A35_izq = Acondicionar(A35,118,137,160,53,75)
A35_der = Acondicionar(A35,118,137,160,105,127)
A36 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/314_116.nii')).get_fdata()
A36_izq = Acondicionar(A36,116,139,160,53,75)
A36_der = Acondicionar(A36,116,140,160,105,125)
A37 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/315_106.nii')).get_fdata()
A37_izq = Acondicionar(A37,106,129,154,51,75)
A37_der = Acondicionar(A37,106,130,154,100,123)
A38 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/316_120.nii')).get_fdata()
A38_izq = Acondicionar(A38,120,139,154,50,75)
A38_der = Acondicionar(A38,120,140,157,100,128)
A39 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/325_118.nii')).get_fdata()
A39_izq = Acondicionar(A39,118,139,154,55,78)
A39_der = Acondicionar(A39,118,140,157,100,123)
A40 = (nibabel.load('C:/ITBA/2021/PIB/TP Cuatrimestral/Imágenes/Alzheimer/326_122.nii')).get_fdata()
A40_izq = Acondicionar(A40,122,137,154,52,78)
A40_der = Acondicionar(A40,122,138,157,100,125)
'''

###TERMINO DE CARGAR IMAGENES###

input = []
output = []
#F=1,M=2
#CN=1,AD=2

izq1s = Metricas(S1_izq,0)
der1s = Metricas(S1_der,0)
prom1s = ArmarProm(izq1s,der1s)
input1s = ArmarInput(prom1s,65.15,1,28)
input.append(input1s)
output.append(1)
izq2s = Metricas(S2_izq,0)
der2s = Metricas(S2_der,0)
prom2s = ArmarProm(izq2s,der2s)
input2s = ArmarInput(prom2s,48.03,1,29)
input.append(input2s)
output.append(1)
izq3s = Metricas(S3_izq,0)
der3s = Metricas(S3_der,0)
prom3s = ArmarProm(izq3s,der3s)
input3s = ArmarInput(prom3s,62.09,2,27)
input.append(input3s)
output.append(1)
izq4s = Metricas(S4_izq,0)
der4s = Metricas(S4_der,0)
prom4s = ArmarProm(izq4s,der4s)
input4s = ArmarInput(prom4s,71.53,2,29)
input.append(input4s)
output.append(1)
izq5s = Metricas(S5_izq,0)
der5s = Metricas(S5_der,0)
prom5s = ArmarProm(izq5s,der5s)
input5s = ArmarInput(prom5s,68.39,1,30)
input.append(input5s)
output.append(1)
izq6s = Metricas(S6_izq,0)
der6s = Metricas(S6_der,0)
prom6s = ArmarProm(izq6s,der6s)
input6s = ArmarInput(prom6s,70.35,1,30)
input.append(input6s)
output.append(1)
izq7s = Metricas(S7_izq,0)
der7s = Metricas(S7_der,0)
prom7s = ArmarProm(izq7s,der7s)
input7s = ArmarInput(prom7s,65.26,2,30)
input.append(input7s)
output.append(1)
izq8s = Metricas(S8_izq,0)
der8s = Metricas(S8_der,0)
prom8s = ArmarProm(izq8s,der8s)
input8s = ArmarInput(prom8s,71.02,2,29)
input.append(input8s)
output.append(1)
izq9s = Metricas(S9_izq,0)
der9s = Metricas(S9_der,0)
prom9s = ArmarProm(izq9s,der9s)
input9s = ArmarInput(prom9s,64.81,1,27)
input.append(input9s)
output.append(1)
izq10s = Metricas(S10_izq,0)
der10s = Metricas(S10_der,0)
prom10s = ArmarProm(izq10s,der10s)
input10s = ArmarInput(prom10s,79.38,1,29)
input.append(input9s)
output.append(1)


izq1a = Metricas(A1_izq,0)
der1a = Metricas(A1_der,0)
prom1a = ArmarProm(izq1a,der1a)
input1a = ArmarInput(prom1a,76.64,1,27)
input.append(input1a)
output.append(2)
izq2a = Metricas(A2_izq,0)
der2a = Metricas(A2_der,0)
prom2a = ArmarProm(izq2a,der2a)
input2a = ArmarInput(prom2a,80.13,1,26)
input.append(input2a)
output.append(2)
izq3a = Metricas(A3_izq,0)
der3a = Metricas(A3_der,0)
prom3a = ArmarProm(izq3a,der3a)
input3a = ArmarInput(prom3a,71.94,1,25)
input.append(input3a)
output.append(2)
izq4a = Metricas(A4_izq,0)
der4a = Metricas(A4_der,0)
prom4a = ArmarProm(izq4a,der4a)
input4a = ArmarInput(prom4a,81.42,1,27)
input.append(input4a)
output.append(2)
izq5a = Metricas(A5_izq,0)
der5a = Metricas(A5_der,0)
prom5a = ArmarProm(izq5a,der5a)
input5a = ArmarInput(prom5a,68.39,2,27)
input.append(input5a)
output.append(2)
izq6a = Metricas(A6_izq,0)
der6a = Metricas(A6_der,0)
prom6a = ArmarProm(izq6a,der6a)
input6a = ArmarInput(prom6a,78.91,2,29)
input.append(input6a)
output.append(2)
izq7a = Metricas(A7_izq,0)
der7a = Metricas(A7_der,0)
prom7a = ArmarProm(izq7a,der7a)
input7a = ArmarInput(prom7a,75.81,1,29)
input.append(input7a)
output.append(2)
izq8a = Metricas(A8_izq,0)
der8a = Metricas(A8_der,0)
prom8a = ArmarProm(izq8a,der8a)
input8a = ArmarInput(prom8a,68.71,2,25)
input.append(input8a)
output.append(2)
izq9a = Metricas(A9_izq,0)
der9a = Metricas(A9_der,0)
prom9a = ArmarProm(izq9a,der9a)
input9a = ArmarInput(prom9a,70.05,1,27)
input.append(input9a)
output.append(2)
izq10a = Metricas(A10_izq,0)
der10a = Metricas(A10_der,0)
prom10a = ArmarProm(izq10a,der10a)
input10a = ArmarInput(prom10a,72.67,1,22)
input.append(input10a)
output.append(2)

#print(input)
#print(output)

x = SelectKBest(chi2, k=4).fit_transform(np.abs(input), output)

print(x)