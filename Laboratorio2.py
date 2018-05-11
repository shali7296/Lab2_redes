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
    
def ifftGraphic(data, rate, fftData,title): # Aquí me deje llevar, no se porqie no funciona :(
	datas = len(data)
	duration = datas/float(rate)
	t = linspace(0, duration, datas)
	ifftData = ifft(fftData, datas)
	plt.plot(t,ifftData*len(fftData)) #Aquí teniamos t,t -> por eso graficaba una recta :o
	plt.title(title)
	plt.xlabel("Tiempo [s]")
	plt.ylabel("Amplitud [dB]")
	plt.savefig("graphics/" + title + ".png")
	plt.close('all')
	return
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

"""
Función general utilizada para graficar.
Entrada: 
		title -> titulo del gráfico
		xlabel -> nombre del eje x
		xdata -> datos del eje x
		ylabel -> nombre del eje y 
		ydata -> datos del eje y
"""
def makeGraphic(title, xlabel, xdata, ylabel, ydata):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata)
    #plt.show()
    plt.savefig("graphics/" + title + ".png")
    plt.close('all')

"""
Función que realiza el filtro paso bajo sobre una señal dada, eliminando todas las frecuencias altas y manteniendo las bajas.
Entrada: 
		data -> señal en dominio del tiempo.
        rate -> frecuencia de muestreo de la señal. 
Salida:
		Señal filtrada
"""
def lowFilter(data,rate):
	nyq = rate / 2
	cutoff = 1000
	numtaps = cutoff + 1
	coeff = firwin(numtaps,(cutoff/nyq)) #,window= 'blackmanharris'
	filtered = lfilter(coeff,1.0,data)
	return filtered

"""
Función que realiza el filtro paso alto sobre una señal dada, eliminando todas las frecuencias bajas y manteniendo las altas.
Entrada: 
		data -> señal en dominio del tiempo.
        rate -> frecuencia de muestreo de la señal. 
Salida:
		Señal filtrada
"""
def highFilter(data,rate):
	nyq = rate / 2
	cutoff = 3000
	numtaps = cutoff + 1
	coeff = firwin(numtaps,(cutoff/nyq),pass_zero = False)
	filtered = lfilter(coeff,1.0,data)
	return filtered

"""
Función que realiza el filtro paso banda sobre una señal dada, que consiste en mantener las frecuencias dentro de un rango definido,
eliminando las frecuencias superiores e inferiores a ese rango.
Entrada: 
		data -> señal en dominio del tiempo.
        rate -> frecuencia de muestreo de la señal. 
Salida:
		Señal filtrada
"""
def bandFilter(data,rate): 	
	nyq = rate / 2
	cutoff_low = 1300
	cutoff_high = 7500
	numtaps = 1201
	coeff_low = firwin(numtaps,(cutoff_low/nyq))
	coeff_high = firwin(numtaps,(cutoff_high/nyq),pass_zero = False)
	bandFilter =- (coeff_low+coeff_high)
	bandFilter[numtaps/2] = bandFilter[numtaps/2] + 1
	filtered = lfilter(bandFilter,1.0,data)
	return filtered

def timeGraphic(data, rate,title):
	duration = len(data)/rate # Tiempo que dura todo el audio
	t = linspace(0, duration, len(data)) # Intervalos de tiempo de 0 a t, generando la misma cantidad de datos que hay en data o vector tiempo
	makeGraphic(title, "Tiempo [s]", t, "Amplitud [dB]", data)
	return 

###################################################
################ Bloque Principal #################
###################################################
#Lectura del archivo .wav
rate, data, times = openWav("audios/beacon.wav") #rate = frecuencia, data = tiempo

#A continuación, se grafica el espectrograma, la transformada de fourier y la antitransformada, para finalmente transformar en un archivo .wav.
#Audio sin filtro:
graphicSpectrogram(data, rate, "Espectrograma")
timeGraphic(data,rate,"Grafico tiempo senal original")
fftData, fftFreqs = tFourier(data,rate)
y = linspace(0, rate, len(abs(fftData)))
makeGraphic("Grafico frecuencia senal original", "Frecuencia [Hz]", y, "Amplitud [dB]", abs(fftData))
ifftGraphic(data, rate, fftData,"Antitransformada de la senal original")

#Filtro Paso Bajo: Mantiene las frecuencias bajas, eliminando las frecuencias altas. 
lowFiltered = lowFilter(data,rate)
timeGraphic(lowFiltered,rate,"Grafico tiempo con filtro paso bajo")
graphicSpectrogram(lowFiltered, rate, "Espectrograma Filtrado Paso Bajo")
fftData1, fftFreqs1 = tFourier(lowFiltered,rate)
y = linspace(0, rate, len(abs(fftData1)))
makeGraphic("Grafico frecuencia con filtro de paso bajo", "Frecuencia [Hz]", y, "Amplitud [dB]", abs(fftData1))
ifftGraphic(data, rate, fftData1,"Anti transformada de la senal filtro paso bajo")

#ifftData1 = tiFourier(fftData1)
#makeGraphic("Anti transformada de la señal filtro paso bajo", "Tiempo [seg]",ifftData1,"Amplitud [dB]",rate)
saveWav("AudioFiltrado_PasoBajo", rate, lowFiltered)

#Filtro Paso Alto: Mantiene las frecuencias altas, eliminando las frecuencias bajas.
highFiltered = highFilter(data,rate)
timeGraphic(highFiltered,rate,"Grafico tiempo con filtro paso alto")
graphicSpectrogram(highFiltered, rate, "Espectrograma Filtrado Paso Alto")
fftData2, fftFreqs2 = tFourier(highFiltered,rate)
y = linspace(0, rate, len(abs(fftData2)))
makeGraphic("Grafico frecuencia con filtro de paso Alto", "Frecuencia [Hz]", y, "Amplitud [dB]", abs(fftData2))
ifftGraphic(data, rate, fftData2,"Anti transformada de la senal filtro paso alto")

#ifftData2 = tiFourier(fftData2)
#makeGraphic("Anti transformada de la señal filtro paso alto", "Tiempo [seg]",ifftData2,"Amplitud [dB]",rate)
saveWav("AudioFiltrado_PasoAlto", rate, highFiltered)

#Filtro Paso Banda: Mantiene las frecuencias dentro de un rango definido, eliminando sus alrededores.
bandFiltered = bandFilter(data,rate)
timeGraphic(bandFiltered,rate,"Grafico tiempo con filtro paso banda")
graphicSpectrogram(bandFiltered, rate, "Espectrograma Filtrado Paso Banda")
fftData3, fftFreqs3 = tFourier(bandFiltered,rate)
y = linspace(0, rate, len(abs(fftData3)))
makeGraphic("Grafico frecuencia con filtro de paso Banda", "Frecuencia [Hz]", y, "Amplitud [dB]", abs(fftData3))
ifftGraphic(data, rate, fftData3,"Anti transformada de la senal filtro paso banda")

#ifftData3 = tiFourier(fftData3)
#makeGraphic("Anti transformada de la señal filtro paso banda", "Tiempo [seg]",ifftData3,"Amplitud [dB]",rate)
saveWav("AudioFiltrado_PasoBanda", rate, bandFiltered)

print("Ejecucion Terminada.")



"""
- PasoBajo: 1000Hz
- PasoAlto: 2500 o 3000
- PasoBanda: 1300 - 3000 Hz
- Aplicar el shift para rectificar
- Aplicar anti-transformada
- Se debe mostrar por pantalla, o guardarla como gráfico basta?
"""