import copy
import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, lfilter
from scipy.signal import welch
from scipy.signal import butter, find_peaks

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

# FRESOLUTION STANDARDIZATION
def resize_frame(frame, target_width, target_height):
    return cv2.resize(frame, (target_width,target_height), interpolation=cv2.INTER_LINEAR)

# FRAMERATE STANDARDIZATION
def interpolate_frames(frames, target_frame_rate, current_frame_rate=1):
    ratio = target_frame_rate / current_frame_rate
    if ratio > 1:
        # Increase the frame rate by interpolating frames
        interpolated_frames = []
        for i in range(len(frames) - 1):
            interpolated_frames.append(frames[i])
            for j in range(1, int(ratio)):
                interpolated_frame = cv2.addWeighted(frames[i], 1 - j / ratio, frames[i + 1], j / ratio, 0)
                interpolated_frames.append(interpolated_frame)
        interpolated_frames.append(frames[-1])
        return interpolated_frames
    elif ratio < 1:
        # Decrease the frame rate by selecting a subset of frames
        reduced_frames = []
        step = int(1 / ratio)
        for i in range(0, len(frames), step):
            reduced_frames.append(frames[i])
        return reduced_frames
    else:
        # If the ratio is 1, return the original frames
        return frames

# CONVERT COLOR SPACE
def convert_color_space(frame, target_color_space=cv2.COLOR_BGR2RGB):
    return cv2.cvtColor(frame, target_color_space)

# DYNAMIC RANGE ADJUSTMENT
def adjust_dynamic_range(frame):
    return cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))

# INTENSITY NORMALIZATION
def normalize_intensity(frame):
    return frame/255.0

# LENS DISTORTION CORRECTION
def correct_lens_distortion(frame, camera_matrix, dist_coeffs):
    return cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)

# CAMERA CALIBRATION
def calibrate_camera(frames, pattern_size):
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    if not objpoints or not imgpoints:
        raise ValueError("Calibration failed: No chessboard corners detected in the calibration frames.")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return camera_matrix, dist_coeffs

class StandardizeFrame:

    def __init__(self, target_resolution, target_frame_rate, target_color_space, pattern_size=(9,6)):

        self.target_resolution = target_resolution
        self.target_frame_rate = target_frame_rate
        self.target_color_space = target_color_space

        self.pattern_size = pattern_size

    def standardize(self, frames, current_frame_rate):

        # RESOLUTION STANDARDIZATION
        frames = [resize_frame(frame, target_width=self.target_resolution[0], target_height=self.target_resolution[1]) for frame in frames]

        # FRAMERATE STANDARDIZATION
        frames = interpolate_frames(frames, self.target_frame_rate, current_frame_rate=current_frame_rate)

        # # CONVERT COLORSPACE
        # frames = [convert_color_space(frame,self.target_color_space) for frame in frames]

        # INTENSITY NORMALIZATION
        frames = [normalize_intensity(frame) for frame in frames]

        return frames
    
    def calibrate(self, calibration_frames):
        self.camera_matrix, self.dist_coefs = calibrate_camera(calibration_frames, self.pattern_size)

def detect_faces(frame, face_cascade_=face_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces
    
def hsv_extraction(frame):
    frame_copy = frame.copy()
    frame_copy = convert_color_space(frame_copy)
    return cv2.cvtColor(frame_copy, cv2.COLOR_RGB2HSV)
    
def skin_params(alpha,frame,filter_size):
    s_values = frame[:,:,1].flatten()
    hist, s_values = np.histogram(s_values)
    hist = median_filter(hist,size=filter_size)
    hist_max = s_values[np.argmax(hist)]
    THrange = alpha*hist_max
    return hist_max, THrange

def extract_skin(alpha,frame,hsv_frame,filter_size=3):
    hist_max, THrange = skin_params(alpha,hsv_frame,filter_size)
    indices_lesser = np.where(hsv_frame[:,:,1]<(hist_max-int(0.5*THrange)))
    frame[indices_lesser[0],indices_lesser[1],:] -= (frame[indices_lesser[0],indices_lesser[1],:]*0.8).astype(np.uint8)
    indices_greater = np.where(hsv_frame[:,:,1]>(hist_max+int(0.5*THrange)))
    frame[indices_greater[0],indices_greater[1],:] -= (frame[indices_greater[0],indices_greater[1],:]*0.8).astype(np.uint8)
    return frame, len(indices_lesser[0])+len(indices_greater[0])

class FaceDetection:

    def __init__(self):
        pass

    def extract(self,frames):

        # DETECT FACE
        faces = [detect_faces(frame) for frame in frames]
                
        return faces, frames
    
class PPGIcomputation:

    def __init__(self,alpha,filter_size,target_frame_size=(640,480)):
        self.alpha = alpha
        self.filter_size = filter_size
        self.target_width = self.W = target_frame_size[0]
        self.target_height = self.H = target_frame_size[1]

    def extract_face(self,frames):

        # CONVERT FROM RGB TO HSV
        hsv_frames = []
        for frame in frames:
            hsv_frames.append(hsv_extraction(frame))
            assert not np.array_equal(frame, hsv_frames[-1])

        # EXTRACT FACIAL SKIN ALONE
        facial_frames = []
        Ck_s = []
        for frame,hsv_frame in zip(frames,hsv_frames):
            facial_frame, ck = extract_skin(self.alpha,frame,hsv_frame,self.filter_size)
            facial_frames.append(facial_frame)
            Ck_s.append(ck)

        return facial_frames, Ck_s
    
    def compute_ppgi(self,frames,Ck_s):

        # COMPUTE AVERAGE PIXEL VALUE
        frames_pixels_average = []
        for frame, ck in zip(frames,Ck_s):
            frame_pixel_average = np.sum(np.sum(frame,axis=1),axis=0)/(((self.W*self.H) - ck)+1E-2)
            frames_pixels_average.append(frame_pixel_average)

        return frames_pixels_average
    
    def compute_ppg(self,frames_pixels_average):
        
        P = frames_pixels_average
        R = P[:,0]
        G = P[:,1]
        B = P[:,2]

        X = 3*R - 2*G
        Y = 1.5*R - G -1.5*B
        
        beta = np.std(X)/np.std(Y)

        S = X - beta*Y

        return S
    
class HRcompute:

    def __init__(self,lowcut,highcut,filter_order,frame_rate=30):
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.fs = frame_rate

    def compute(self,ppg):

        ppg = butter_bandpass_filter(ppg, self.lowcut, self.highcut, self.fs, order=self.filter_order)

        segment_length = int(len(ppg)/8)
        segments = []
        for i in range(7):
            segment = ppg[i*segment_length:((i+1)*segment_length)+int(segment_length/2)]
            window = np.hamming(len(segment))
            segment = window*segment
            segments.append(segment)
        segment = ppg[7*segment_length:]
        window = np.hamming(len(segment))
        segment = window*segment
        segments.append(segment)

        segments_psd = []
        for segment in segments:
            frequencies, psd = welch(segment, fs=self.fs, window='hamming')
            segments_psd.append((frequencies,psd))

        fh = 0
        for psd in segments_psd:
            fh += psd[0][np.argmax(psd[1])]*60
        heart_rate = fh/len(segments_psd)

        return ppg, segments, segments_psd, heart_rate
    
class HRcompute2:

    def __init__(self,lowcut,highcut,filter_order,frame_rate=30):
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.fs = frame_rate

    def compute(self,ppg):

        ppg = butter_bandpass_filter(ppg, self.lowcut, self.highcut, self.fs, order=self.filter_order)
        window = np.hamming(len(ppg))
        ppg = window*ppg
        frequencies, psd = welch(ppg, fs=self.fs, window='hamming')
        psd = psd/psd.sum()
        heart_rate = (psd.dot(frequencies))*60
        return heart_rate
    
    def get_psd(self,ppg):
        
        ppg = butter_bandpass_filter(ppg, self.lowcut, self.highcut, self.fs, order=self.filter_order)
        window = np.hamming(len(ppg))
        ppg = window*ppg
        frequencies, psd = welch(ppg, fs=self.fs, window='hamming')

        return frequencies, psd

    def compute_from_psd(self, frequencies, psd):
        psd = psd/psd.sum()
        heart_rate = (psd.dot(frequencies))*60
        return heart_rate

class ibi_HRcompute:

    def __init__(self,lowcut,highcut,filter_order,frame_rate=30):
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.fs = frame_rate

    def compute(self,ppg):
        ppg = butter_bandpass_filter(ppg, self.lowcut, self.highcut, self.fs, order=self.filter_order)
        
        ppg = butter_bandpass_filter(
            ppg,
            self.lowcut, self.highcut,
            self.fs, self.filter_order
        )

        # Compute peak locations
        ppg = (ppg-ppg.min())/(ppg.max()-ppg.min())
        peaks, _ = find_peaks(ppg, height=0.5*ppg.max())

        # if len(peaks<=1):
        #     if len(peaks)==1:
        #         peaks = np.append(peaks,peaks[0])
        #     else:
        #         peaks = np.append(peaks,)

        # Compute heart rate
        beat_to_beat_interval = np.diff(peaks) / self.fs
        heart_rate = 60 / (np.mean(beat_to_beat_interval)+0.001)

        if np.isnan(heart_rate):
            print(peaks,beat_to_beat_interval)

        return heart_rate