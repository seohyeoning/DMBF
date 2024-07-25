import os
import numpy as np
import math
import scipy.io as sio
from scipy.fftpack import fft,ifft

"""
https://github.com/ziyujia/Signal-feature-extraction_DE-and-PSD/blob/master/DE_PSD.py
"""

def DE_PSD(args,data):
    '''
    compute DE and PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate
            stft_para.fStart    start frequency of each frequency band
            stft_para.fEnd      end frequency of each frequency band
            stft_para.window    window length of each sample point(seconds)
            stft_para.fs        original frequency
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    
    fStart=[1, 4, 8, 12, 30]
    fEnd=[4, 8, 12, 30, 45]
    fs=args.sr
    window=args.window_size
    WindowPoints=fs*window
    STFTN=512

    fStartNum=np.zeros([len(fStart)],dtype=int)
    fEndNum=np.zeros([len(fEnd)],dtype=int)
    for i in range(0,len(fStart)):
        fStartNum[i]=int(fStart[i]/fs*STFTN)
        fEndNum[i]=int(fEnd[i]/fs*STFTN)

    #print(fStartNum[0],fEndNum[0])
    s, n, m = data.shape  # 샘플 수, 채널 수, 시간 포인트 수

    #print(m,n,l)
    psd = np.zeros([s,n,len(fStart)])
    de = np.zeros([s,n,len(fStart)])
    #Hanning window
    Hlength=window*fs
    #Hwindow=hanning(Hlength)
    Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])

    for i in range(s):  # 각 샘플에 대해 반복
        for j in range(n):  # 각 채널에 대해 반복
            temp = data[i, j] * Hwindow
            FFTdata = fft(temp, STFTN)
            magFFTdata = np.abs(FFTdata[:STFTN // 2])

            for p in range(len(fStart)):
                E = np.sum(magFFTdata[fStartNum[p]-1:fEndNum[p]] ** 2)
                E /= (fEndNum[p] - fStartNum[p] + 1)
                psd[i, j, p] = E
                # de[i, j, p] = math.log(100 * E, 2)
    
    return psd

