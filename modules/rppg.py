import math
import numpy as np
from scipy.signal import savgol_filter, find_peaks, welch
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import norm
from scipy.ndimage import median_filter
from . filters import butter_bandpass_filter, apply_notch_filter, cheby2_bandpass_filter
import colorsys
import cv2

def landmark_extraction(frame, landmarks, indices, width, height):
    roi_ = []
    for idx in indices:
        x = int(landmarks.landmark[idx].x * width)
        y = int(landmarks.landmark[idx].y * height)
        if x<=width-1 and y<=height-1:
            roi = frame[y,x,:]
            roi_.append(roi)
    return np.array(roi_)

def ppgi_2_ppg(ppgi, f1, f2, fs, order_, fn, Q):

    w = int(0.2*len(ppgi))
    
    R = ppgi[:,2]/ppgi[:,2].mean()
    G = ppgi[:,1]/ppgi[:,1].mean()
    B = ppgi[:,0]/ppgi[:,0].mean()
    
    R = cheby2_bandpass_filter(R, f1, f2, fs, Q, order_)
    G = cheby2_bandpass_filter(G, f1, f2, fs, Q, order_)
    B = cheby2_bandpass_filter(B, f1, f2, fs, Q, order_)


    X = 3*R - 2*G
    Y = 1.5*R - G -1.5*B
    X = X - X.mean()
    Y = Y - Y.mean()
    beta = np.std(X)/np.std(Y)
    S = X - beta*Y

    # S = G

    # S = G/R

    S = 0.6*np.log(1+G) + 0.4*S

    # S = 0.2*np.convolve(np.log(1+G),S,mode="same") +0.8*np.log(1+G)

    S = standardize_sig(S)

    return S

def standardize_sig(S):
    # S = (S-S.mean())/S.std()
    S = (S-S.min())/(S.max()-S.min())
    S = S * 2 - 1
    return S

def min_max_sig(S):
    S = (S-S.min())/(S.max()-S.min())
    return S

def compute_landmark_ppgi(landmark_data):

    ppgi_ = []
    for landmark in landmark_data:
        if landmark.ndim==3:
            ROI = landmark.shape[0]*landmark.shape[1]
            ldmk = np.mean(landmark, axis=(0,1))/ROI
        elif landmark.ndim==2:
            ROI = landmark.shape[0]
            ldmk = np.mean(landmark, axis=0)/ROI
        ppgi_.append(ldmk)
    ppgi_ = np.array(ppgi_)

    return ppgi_/np.mean(ppgi_, axis=0)

def savitzky_golay(signal, window_length, poly_order):
    return savgol_filter(signal, window_length, poly_order)

def gaussian_filter(signal, sigma):
    return gaussian_filter1d(signal, sigma)

def FFT_BPF(signal, ls, hs, fs):
    
    n = len(signal)
    
    fft_values = fft(signal)
    fft_freqs = fftfreq(n, 1 / fs)
    
    low_cut = ls
    high_cut = hs
    
    filter_mask = (np.abs(fft_freqs) >= low_cut) & (np.abs(fft_freqs) <= high_cut)
    
    filtered_fft_values = fft_values * filter_mask
    
    filtered_signal = ifft(filtered_fft_values)
    
    return filtered_signal

def harmonic_2nd_HR(ppg, fs):
    n = len(ppg)
    fft_values = fft(ppg)
    fft_freqs = fftfreq(n, 1/fs)
    
    bpm_freqs = fft_freqs * 60
    lower_bound_bpm = 240
    upper_bound_bpm = 360

    
    mask = (bpm_freqs >= lower_bound_bpm) & (bpm_freqs <= upper_bound_bpm)
    filtered_fft_values = np.abs(fft_values[mask])
    filtered_bpm_freqs = bpm_freqs[mask]

    hr_index = np.where(filtered_fft_values>0.7*filtered_fft_values.max())
    filtered_fft_values = filtered_fft_values[hr_index]
    filtered_fft_values = filtered_fft_values/filtered_fft_values.sum()
    filtered_bpm_freqs = filtered_bpm_freqs[hr_index]
    hr_ = filtered_fft_values.dot(filtered_bpm_freqs)
    heart_rate = hr_ / 3
    
    return heart_rate

def EFA(ppg, fps, sigma_bpm):
    ppg = ppg - ppg.mean()
    ppg = butter_bandpass_filter(ppg, 1, 3, fps, order=sigma_bpm)
    window = np.hamming(len(ppg))
    ppg = window*ppg
    frequencies, psd = welch(ppg, fs=fps, window='hamming')
    sigma = sigma_bpm / 60
    psd = gaussian_filter1d(psd, sigma * len(psd) / (fps / 2))
    psd = psd/psd.sum()
    heart_rate = (psd.dot(frequencies))*60
    
    return heart_rate

def welch_HR(ppg, fps):
    window = np.hamming(len(ppg))
    ppg = window*ppg
    frequencies, psd = welch(ppg, fs=fps, window='hamming')
    # psd = psd/psd.sum()
    # heart_rate = (psd.dot(frequencies))*60
    indices = np.argmax(psd)
    heart_rate = frequencies[indices]*60
    
    return heart_rate

def median_analysis(signal, fps):
    signal = signal - signal.mean()
    peaks, _ = find_peaks(signal)
    distances = np.diff(peaks)
    min_distance = 0.2 * fps
    max_distance = 3.0 * fps
    filtered_distances = distances[(distances >= min_distance) & (distances <= max_distance)]
    if len(filtered_distances) == 0:
        return np.nan  # Return NaN if no distances are within the specified range
    median_distance = np.median(filtered_distances)
    hr = (fps*60)/(median_distance)

    return hr

def ibi_HR(signal, fps):
    signal = (signal-signal.min())/(signal.max()-signal.min())
    peaks, _ = find_peaks(signal, height=0.5*signal.max())

    beat_to_beat_interval = np.diff(peaks) / fps
    heart_rate = 60 / (np.mean(beat_to_beat_interval))

    if np.isnan(heart_rate):
        heart_rate = 0

    return heart_rate

# Gaussian kernel function
def gaussian_kernel(x):
    return norm.pdf(x)

def parzen_rosenblatt_window(hr_values, bandwidth):
    
    n = len(hr_values)
    smoothed_hr = np.zeros(n)
    
    # Loop over each HR value
    for i in range(n):
        weights = np.zeros(n)
        for j in range(n):
            weights[j] = gaussian_kernel((hr_values[i] - hr_values[j]) / bandwidth)
        
        # Normalize weights
        weights /= np.sum(weights)
        
        # Compute the smoothed HR value
        smoothed_hr[i] = np.sum(weights * hr_values)
    
    return smoothed_hr

def rgb_to_saturation(rgb_list):
    saturation_values = []
    
    for rgb in rgb_list:
        r, g, b = rgb
        r /= 255.0
        g /= 255.0
        b /= 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        saturation_values.append(s)
    
    return np.array(saturation_values)

def measure_s(frame):
    if frame.ndim==3:
        frame = frame.reshape(frame.shape[0]*frame.shape[1], frame.shape[2])
    s_values = rgb_to_saturation(frame)
    hist, s_values = np.histogram(s_values)
    hist = median_filter(hist,size=5)
    hist_max = s_values[np.argmax(hist)]
    return hist_max

def area_measure_s(area):
    s_values = []
    for frame_ in area:
        s_values.append(measure_s(frame_))
    return np.array(s_values).mean()