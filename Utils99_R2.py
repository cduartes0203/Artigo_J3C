
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from scipy.stats import entropy, kurtosis, entropy
from scipy.signal import hilbert, chirp
import os
import pickle as pkl
import re
import scipy.stats as stats
import math
from scipy import signal
from scipy.signal import savgol_filter, stft
from scipy.ndimage import gaussian_filter1d, median_filter
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

fs=32000

def signal_entropy(signal, bins=50):
    """Calcula a entropia de um sinal transformando-o em distribuição de probabilidades."""
    
    # Criar histograma normalizado (estimativa de distribuição de probabilidades)
    hist, _ = np.histogram(signal, bins=bins, density=True)
    
    # Remover valores zero para evitar log(0)
    hist = hist[hist > 0]

    # Calcular entropia de Shannon
    return entropy(hist, base=3)  # Base 2 para entropia em bits (opcional)

def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())

# filters ###############################################################################################################################

def moving_average_df(df, window_size):

    return df.apply(lambda col: np.convolve(col, np.ones(window_size) / window_size, mode='same'))

def exponential_moving_average(data, alpha=0.3):
    ema = [data.iloc[0]]  # Start with the first value
    for i in range(1, len(data)):
        ema.append(alpha * data.iloc[i] + (1 - alpha) * ema[i-1])
    return pd.Series(ema, index=data.index)

def exponential_moving_average_df(df, alpha=0.3):
    ema_df = df.copy()
    for col in df.columns:
        ema_df[col] = exponential_moving_average(df[col], alpha)
    return ema_df

# features ###############################################################################################################################

def calculate_entropy(vector):
    contagem = Counter(vector)
    total = len(vector)
    # Calcular a probabilidade de cada elemento
    probabilidades = np.array([freq / total for freq in contagem.values()])
    # Aplicar a fórmula da entropia de Shannon
    entropia = -np.sum(probabilidades * np.log10(probabilidades))
    
    return entropia

def calculate_crest_factor(vector):
    rms = np.sqrt(np.mean(np.square(vector)))
    peak = np.max(np.abs(vector))
    return peak / rms

def calculate_wave_factor(vector):
    rms = np.sqrt(np.mean(np.square(vector)))
    mean = np.mean(np.abs(vector))
    return rms / mean

def calculate_impulse_factor(vector):
    peak = np.max(np.abs(vector))
    mean = np.mean(np.abs(vector))
    return peak / mean

def calculate_margin_factor(vector):
    peak = np.max(np.abs(vector))
    rms = np.sqrt(np.mean(np.square(vector)))
    return peak / (rms ** 2)

def time_features(vec):
    vec =vec - np.mean(vec)
    mn = (np.mean(vec))
    mn_abs = (np.mean(np.abs(vec)))
    var = (np.var(vec))
    std = math.sqrt(var)
    skw = (stats.skew(vec))
    krt = (stats.kurtosis(vec))
    ntrpy = signal_entropy(vec)
    rms = (np.sqrt(np.mean(np.square(np.array(vec)))))
    max = np.max(vec)
    ptp = (np.ptp(vec))

    features = [mn,mn_abs,var,std,skw,krt,ntrpy,rms,max,ptp]
    column_names =['Média','Média Absoluta','Variância','Desvio Parão',
                   'Assimetria','Kurtose','Entropia','RMS','Máximo','Pico-a-pico']
    return features, column_names
def freq_features(vec,rate=fs):
    vec = vec - np.mean(vec)
    freq, fft1 = envelope(vec,rate)
    fft = fft1[(freq>=1) & (freq<=10000)].copy()
    freq = freq[(freq>=1) & (freq<=10000)].copy()
    energy = np.sum(fft**2)
    mn_energy = (np.mean(fft**2))
    var_energy = (np.var(fft**2))
    std_energy = math.sqrt(var_energy)
    skw_energy= (stats.skew(fft**2))
    ntrpy = signal_entropy(fft**2)
    krt_energy = (stats.kurtosis(fft**2))
    energy_density = np.sum((fft)**2)/np.sum(fft1**2)
    rms_energy = (np.sqrt(np.mean(np.square(np.array(fft**2)))))

    features = [energy,mn_energy,var_energy,std_energy,
                skw_energy,ntrpy,krt_energy,energy_density,rms_energy]
    column_names =['Energia','Energia Média','Variância da Energia','Desvio Padrão da energia'
                   ,'Assimeria da Energia','Entropia da Energia','Kurtose da Energia','Densidade da Energia','RMS da Energia']
    return features, column_names


# fft calculation ########################################################################################################################
def fft_calc(signal,rate):
  fft_shft_abs = np.fft.fftshift(2.0 * np.abs(np.fft.fft(signal) / np.size(np.fft.fft(signal))))
  freq_shft = np.fft.fftshift(np.fft.fftfreq(np.size(fft_shft_abs), 1/rate))
  return freq_shft[(freq_shft>=0) & (freq_shft<=rate/2)], fft_shft_abs[(freq_shft>=0) & (freq_shft<=rate/2)]

def df_fft(df,rate):
    fft = df.copy()
    for i in range(len(fft.columns)):
        freq_shft, fft_shft_abs = fft_calc(fft.iloc[:,i], rate)
        fft.iloc[:,i] = fft_shft_abs

    fft.insert(0, 'FREQ',freq_shft)
    return fft

# envelope ########################################################################################################################

def envelope(signal, rate):
  amplitude_envelope = np.abs(hilbert(signal))
  amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
  fft_hilbert = np.fft.fftshift(2.0 * np.abs(np.fft.fft(amplitude_envelope) / np.size(np.fft.fft(amplitude_envelope))))
  freq_hilbert = np.fft.fftshift(np.fft.fftfreq(np.size(fft_hilbert), 1/rate))
  return freq_hilbert[(freq_hilbert>=0) & (freq_hilbert<=rate/2)], fft_hilbert[(freq_hilbert>=0) & (freq_hilbert<=rate/2)]  


def df_env(df,rate):
    env = df.copy()
    for i in range(len(env.columns)):
        freq_shft, fft_shft_abs = envelope(env.iloc[:,i], rate)
        env.iloc[:,i] = fft_shft_abs

    env.insert(0, 'FREQ',freq_shft)
  
    return env

# FILTER ########################################################################################################################

def bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    """
    Aplica um filtro passa-banda em um sinal.

    Parâmetros:
    - signal_data: np.array -> Sinal de entrada
    - lowcut: float -> Frequência de corte inferior (Hz)
    - highcut: float -> Frequência de corte superior (Hz)
    - fs: float -> Frequência de amostragem (Hz)
    - order: int -> Ordem do filtro (padrão: 4)

    Retorna:
    - Sinal filtrado
    """

    nyquist = 0.5 * fs  # Frequência de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist

    # Criando filtro Butterworth passa-banda
    b, a = signal.butter(order, [low, high], btype='band')

    # Aplicando filtro no sinal
    filtered_signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal

# metrics ########################################################################################################################
