import cv2
import numpy as np

# FRESOLUTION STANDARDIZATION
def resize_frame(frame, target_width, target_height):
    return cv2.resize(frame, (target_width,target_height), interpolation=cv2.INTER_LINEAR)

# FRAMERATE STANDARDIZATION
def interpolate_frames(frames, target_frame_rate, current_frame_rate=1):
    if (target_frame_rate/current_frame_rate)==target_frame_rate:
        ratio = target_frame_rate/current_frame_rate
        interpolated_frames = []
        for i in range(len(frames)-1):
            interpolated_frames.append(frames[i])
            for j in range(1, int(ratio)):
                interpolated_frame = cv2.addWeighted(frames[i], 1 - j/ratio, frames[i+1], j/ratio, 0)
                interpolated_frames.append(interpolated_frame)
        interpolated_frames.append(frames[-1])
        return interpolated_frames
    else:
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
def correct_lens_distoetion(frame, camera_matrix, dist_coeffs):
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

class VideoStandardizationPipeline:

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

        # CONVERT COLORSPACE
        frames = [convert_color_space(frame,self.target_color_space) for frame in frames]

        # ADJUST DYNAMIC RANGE
        frames = [adjust_dynamic_range(frame) for frame in frames]

        # INTENSITY NORMALIZATION
        frames = [normalize_intensity(frame) for frame in frames]

        return frames
    
    def calibrate(self, calibration_frames):
        self.camera_matrix, self.dist_coefs = calibrate_camera(calibration_frames, self.pattern_size)