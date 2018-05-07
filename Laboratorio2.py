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

def lowFilter(data,rate):
	nyq = rate / 2
	cutoff = 1000
	numtaps = cutoff + 1
	coeff1 = firwin(numtaps,(cutoff/nyq))
	filtered = lfilter(coeff1,1.0,data)
	return filtered

###################################################
################ Bloque Principal #################
###################################################

#Lectura del archivo .wav
rate, data, times = openWav("audios/beacon.wav") #rate = frecuencia
#Espectrograma del audio sin filtrado.
graphicSpectrogram(data, rate, "Espectrograma")

filtered = lowFilter(data,rate)

graphicSpectrogram(filtered, rate, "Espectrograma Filtrado")

saveWav("AudioFiltrado_PasoBajo", rate, filtered)

print("Exito")


"""La idea es como dice el enunciado es probar distintos parámetros.
Diseñen un filtro paso bajo, otro paso alto y un paso banda.
Deben centrarse en el análisis de cada uno de los filtros y cuál consideran que es mejor y por qué.

En los informes centrense bien en el desarrollo de la experiencia, cómo usan las funciones, por qué 
eligieron los parámetros y de dónde vienen, y el procedimiento. Y analicen en profundidad."""