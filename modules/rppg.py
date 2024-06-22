import numpy as np
from scipy.signal import savgol_filter, find_peaks, welch
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft, fftfreq
from . filters import butter_bandpass_filter, apply_notch_filter

def landmark_extraction(frame, landmarks, indices, width, height):
    roi_ = []
    for idx in indices:
        x = int(landmarks.landmark[idx].x * width)
        y = int(landmarks.landmark[idx].y * height)
        roi = frame[y,x,:]
        roi_.append(roi)
    return np.array(roi_)

def ppgi_2_ppg(ppgi, f1, f2, fs, order_, fn, Q):
    
    R = ppgi[:,2]/ppgi[:,2].mean()
    G = ppgi[:,1]/ppgi[:,1].mean()
    B = ppgi[:,0]/ppgi[:,2].mean()

    R = butter_bandpass_filter(R, f1, f2, fs, order_)
    G = butter_bandpass_filter(G, f1, f2, fs, order_)
    B = butter_bandpass_filter(B, f1, f2, fs, order_)

    # X = 3*R - 2*G
    # Y = 1.5*R - G -1.5*B
    # X = X - X.mean()
    # Y = Y - Y.mean()
    # beta = np.std(X)/np.std(Y)
    # S = X - beta*Y
    S = np.log(1+G)

    # SGR = G/R
    # S = SGR

    # S = 0.8*S + 0.2*SGR

    # S = apply_notch_filter(S, fn, fs, Q)
    # S = butter_bandpass_filter(S, f1, f2, fs, order_)
    
    S = (S-S.mean())/S.std()
    # S = (S-S.min())/(S.max()-S.min())
    S = S * 2 - 1

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

def hr_4rm_2ndHarmonicFourier(ppg, fs):
    n = len(ppg)
    fft_values = fft(ppg)
    fft_freqs = fftfreq(n, 1/fs)
    
    bpm_freqs = fft_freqs * 60
    lower_bound_bpm = 250
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

def median_analysis(signal, fps):
    signal = signal - signal.mean()
    peaks, _ = find_peaks(signal)
    distances = np.diff(peaks)
    min_distance = 0.4 * fps
    max_distance = 1.3 * fps
    filtered_distances = distances[(distances >= min_distance) & (distances <= max_distance)]
    if len(filtered_distances) == 0:
        return np.nan  # Return NaN if no distances are within the specified range
    median_distance = np.median(filtered_distances)
    hr = (fps*60)/(median_distance)

    return hr
    