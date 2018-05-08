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
from numpy import linspace
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, ifft
from scipy.signal import firwin, lfilter

import warnings
warnings.filterwarnings('ignore')


###################################################
############# Definición de funciones #############
###################################################
"""
Función que se encarga de abrir archivos .wav y obtiene la frecuencia e información de la señal.
Entrada:
        name-> nombre del archivo con extensión .wav
Salida:
        rate  -> frecuencia de muestreo.
        info  -> datos de la señal.
        times -> tiempo para cada dato en info.
"""
def openWav(name):
        rate, info = read(name)
        dimension = info[0].size
        if(dimension == 1):
                data = info
        else:
                data = info[:,dimension-1]
        n = len(data)
        Ts = n / rate
        times = np.linspace(0, Ts, n)
        return (rate, data, times)
"""
Función que guarda en un archivo .wav datos de una señal
Entrada:
        title -> nombre de salida del archivo.
        rate  -> frecuencia de muestreo de una señal.
        data  -> señal en dominio del tiempo.
"""
def saveWav(title, rate, data):
        write("audios/" + title + ".wav", rate, data.astype('int16'))

"""
Función que realiza la transformada de fourier en base a los datos obtenidos del audio.
Entrada:
        data     -> señal en dominio del tiempo.
        rate     -> frecuencia de muestreo de la señal.
Salida:
        fftData  -> transformada de fourier normalizada para los valores de la señal original.
        fftFreqs -> frecuencias de muestreo que dependen del largo del arreglo data y de rate.
"""
def tFourier(data,  rate):
        n = len(data)
        Ts = n / rate
        fftData = fft(data) / n
        fftFreqs = np.fft.fftfreq(n, 1/rate)
        return (fftData, fftFreqs)

"""
Función que se encarga de realizar la transformada inversa de fourier a valores que estén en dominio de su frecuencia.
Entrada:
        fftData -> transformada de fourier normalizada.
Salida:
        transformada de fourier inversa desnormalizada, se puede utilizar para escribir archivos .wav (audio)
"""
def tiFourier(fftData):
        return ifft(fftData)*len(fftData)
        



"""
Función que grafica el espectrograma, dado los valores de la señal y el muestreo de frecuencia.
Entrada:
        info -> datos de la señal a la cual se quiere graficar el espectrograma.
        rate ->frecuencia de muestreo de la señal.
        title-> titulo del gráfico.

"""
def graphicSpectrogram(info, rate, title):
	plt.specgram(info,Fs=rate, cmap= 'nipy_spectral', noverlap= 250)
	plt.title(title)
	plt.xlabel('Tiempo [seg]')
	plt.ylabel('Frecuencia [Hz]')
	plt.savefig("graphics/" + title + ".png")
	plt.close('all')

def makeGraphic(title, xlabel, xdata, ylabel, ydata):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata)
    #plt.show()
    plt.savefig("graphics/" + title + ".png")
    plt.close('all')

def lowFilter(data,rate):
	nyq = rate / 2
	cutoff = 1000
	numtaps = cutoff + 1
	coeff = firwin(numtaps,(cutoff/nyq)) #,window= 'blackmanharris'
	filtered = lfilter(coeff,1.0,data)
	return filtered

def highFilter(data,rate):
	nyq = rate / 2
	cutoff = 3000
	numtaps = cutoff + 1
	coeff = firwin(numtaps,(cutoff/nyq),pass_zero = False)
	filtered = lfilter(coeff,1.0,data)
	return filtered

def bandFilter(data,rate):
	return

###################################################
################ Bloque Principal #################
###################################################

#Lectura del archivo .wav
rate, data, times = openWav("audios/beacon.wav") #rate = frecuencia, data = tiempo

#Espectrograma del audio sin filtrado.
graphicSpectrogram(data, rate, "Espectrograma")
fftData, fftFreqs = tFourier(data,rate)
y = linspace(0, rate, len(abs(fftData)))
makeGraphic("Grafico audio sin filtro", "Frecuencia [Hz]", y, "Amplitud [dB]", abs(fftData)) #Si comento esta linea, se ve bien el espectrograma :c

#Filtro Paso Bajo: Mantiene las frecuencias bajas, eliminando las frecuencias altas. 
#Se grafica el espectrograma y se transforma en un archivo .wav.
lowFiltered = lowFilter(data,rate)
graphicSpectrogram(lowFiltered, rate, "Espectrograma Filtrado Paso Bajo")
fftData1, fftFreqs1 = tFourier(lowFiltered,rate)
y = linspace(0, rate, len(abs(fftData1)))
makeGraphic("Grafico audio con filtro de paso bajo", "Frecuencia [Hz]", y, "Amplitud [dB]", abs(fftData1))
saveWav("AudioFiltrado_PasoBajo", rate, lowFiltered)

#Filtro Paso Alto:
highFiltered = highFilter(data,rate)
graphicSpectrogram(highFiltered, rate, "Espectrograma Filtrado Paso Alto")
fftData2, fftFreqs2 = tFourier(highFiltered,rate)
y = linspace(0, rate, len(abs(fftData2)))
makeGraphic("Grafico audio con filtro de paso Alto", "Frecuencia [Hz]", y, "Amplitud [dB]", abs(fftData2))
saveWav("AudioFiltrado_PasoAlto", rate, highFiltered)

#Filtro Paso Alto:
#highFiltered = highFilter(data,rate)
#graphicSpectrogram(highFiltered, rate, "Espectrograma Filtrado Paso Alto")
#fftData2, fftFreqs2 = tFourier(highFiltered,rate)
#y = linspace(0, rate, len(abs(fftData2)))
#makeGraphic("Audio con filtro de paso Alto", "Frecuencia [Hz]", y, "Amplitud [dB]", abs(fftData2))
#saveWav("AudioFiltrado_PasoAlto", rate, highFiltered)
print("Exito")


"""La idea es como dice el enunciado es probar distintos parámetros.
Diseñen un filtro paso bajo, otro paso alto y un paso banda.
Deben centrarse en el análisis de cada uno de los filtros y cuál consideran que es mejor y por qué.

En los informes centrense bien en el desarrollo de la experiencia, cómo usan las funciones, por qué 
eligieron los parámetros y de dónde vienen, y el procedimiento. Y analicen en profundidad."""


"""
- PasoBajo: 1000Hz
- PasoAlto: 2500 o 3000
- PasoBanda: 1300 - 3000 Hz
- Aplicar el shift para rectificar
"""