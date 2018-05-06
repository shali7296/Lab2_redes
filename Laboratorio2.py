"""
Laboratorio 2 de Redes de Computadores por Shalini Ramchandani & Javier Arredondo
 1.	Importe la señal de audio utilizando la función read de scipy.
 2. Mostrar el espectrograma de la función, explicarlo.
 3. Sobre el audio en su dominio de la frecuencia:
		a. Aplique filtro FIR, probar distintos parámetros.
		b. Calcule la transformada de fourier inversa del resultado, compare con la
		   señal original.
		c. Mostrar espectrograma, luego de aplicar el filtro.
 4. Utilizando la función write, guarde los audios del audio filtrado, el audio obtenido en
	la primera experiencia y el audio original, luego compare.
"""

###################################################
################## Importaciones ##################
###################################################
import numpy as np
import matplotlib.pyplot as plt
#from numpy import sin, linspace, pi
from scipy.io.wavfile import read, write
#from scipy import fft

#import warnings
#warnings.filterwarnings('ignore')


###################################################
############# Definición de funciones #############
###################################################

def graphicSpectrogram(info,rate,title):
	plt.specgram(info,Fs=rate)
	plt.title(title)
	plt.xlabel('Time [sec]')
	plt.ylabel('Frecuency')
	plt.savefig(title+".png")
	plt.show()


###################################################
################ Bloque Principal #################
###################################################

#Lectura del archivo .wav
rate, info = read("beacon.wav")
dimension = info[0].size
if (dimension == 1):
	data = info
else:
	data = info[:,dimension-1]

graphicSpectrogram(data,rate,"Espectrograma")