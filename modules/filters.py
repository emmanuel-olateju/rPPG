import numpy as np
from scipy.signal import butter, find_peaks, sosfilt, iirnotch, filtfilt, resample, lfilter

def resample_signal(data, original_fs, target_fs):
    duration = len(data) / original_fs
    new_num_samples = int(duration * target_fs)
    upsampled_signal = resample(data, new_num_samples)
    
    return upsampled_signal

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

# Apply the high-pass filter to data
def butter_highpass_filter(data, cutoff, fs, order=5):
    sos = butter_highpass(cutoff, fs, order)
    y = sosfilt(sos, data)
    return y

def design_notch_filter(freq, fs, Q):
    w0 = freq / (fs / 2)  # Normalize the frequency
    b, a = iirnotch(w0, Q)
    return b, a

def apply_notch_filter(data, freq, fs, Q):
    b, a = design_notch_filter(freq, fs, Q)
    y = filtfilt(b, a, data)
    return y