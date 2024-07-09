import yaml
import numpy as np
import cv2
import mediapipe as mp
from modules import processing
from modules import rppg
from modules import filters
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import time
from scipy.stats import mode
import copy

# Initialize plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# Example 2D numpy array with 3 lines, each with 10 points
data = np.random.rand(3, 10)

channels = ["nose_ppg", "rc_ppg", "face_ppg", "ulip_ppg"]
linestyles = ["dotted", "dotted", "dotted", "dotted"]
lines = []
for i in range(len(channels)):
    line, = ax.plot([], [], linestyle=linestyles[i],label=channels[i])
    lines.append(line)
ax.set_ylim(0, 3)
ax.legend(lines, channels, loc='upper right')
ax.grid(True)

with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
left_cheek_indices = [50, 101, 118, 202, 203, 204, 205, 206, 207, 208, 209, 210]
right_cheek_indices = [280, 351, 437, 418, 417, 416, 415, 414, 413, 412, 411, 410]
nose_indices = [1, 2, 3, 4, 5, 6, 197, 195, 5, 4, 45, 275, 279, 368, 18, 25, 44, 75, 215]
upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
lower_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454]

# right_cheek_indices = right_cheek_indices[0:4] + right_cheek_indices[-3:]
# left_cheek_indices = left_cheek_indices[0:4] + left_cheek_indices[-3:]
# nose_indices = nose_indices[0:4] + nose_indices[-3:]

W = 10
SW = 5
hr_method = rppg.median_analysis

def record():

    HR_estimates = []

    ''' Initialize Video Capture '''
    cap = cv2.VideoCapture(CONFIG["VIDEO"])
    # Check if camers opened succesfully
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CAMERA_FPS = cap.get(cv2.CAP_PROP_FPS)
    print((CAMERA_WIDTH,CAMERA_HEIGHT), CAMERA_FPS)
    CONFIG["TARGET_FRAME_RATE"] = int(CAMERA_FPS)
    print(CONFIG)

    frames = []
    heart_rate = 0
    while True:
        
        ''' 1S FRAME DATA ACQUISITION '''
        # frames = []
        for i in range(int(W*CAMERA_FPS)-len(frames)):
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frames.append(frame)
                if i%int(CAMERA_FPS)==0:
                    cv2.imshow("Video Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        ''' STANDARDIZE FRAMES '''
        # frames = [(frame-frame.min())/(frame.max()-frame.min())*255 for frame in frames]
        # frames = [frame.astype("uint8") for frame in frames]

        ''' DETECT FACE & LANDMARKS '''
        face_frames = []
        right_cheek = []
        nose = []
        ulip = []
        disp_frames = copy.deepcopy(frames)
        for f,frame in enumerate(frames):

            height, width = CAMERA_HEIGHT, CAMERA_WIDTH
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(rgb_frame)

            if face_results.detections:
                if len(face_results.detections)>1:
                    frames = []
                    break
                else:
                    for detection in face_results.detections:
                        # Draw a rectangle around the detected face
                        bboxC = detection.location_data.relative_bounding_box
                        x_min = int(bboxC.xmin * width)
                        y_min = int(bboxC.ymin * height)
                        box_width = int(bboxC.width * width)
                        box_height = int(bboxC.height * height)
                        face_frame = frame[y_min+5:(y_min+box_height)-5,x_min+5:(x_min+box_width)-5,:]
                        face_frames.append(face_frame)
                        # if f>=int((1-((SW/W)/2))*CAMERA_FPS):
                        cv2.rectangle(disp_frames[f], (x_min, y_min), (x_min + box_width, y_min + box_height), (255, 255, 255), 2)
                        text_x = x_min
                        text_y = y_min + box_height + 25
                        cv2.putText(disp_frames[f], f"{heart_rate:.2f} BPM", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            roi_r = rppg.landmark_extraction(frame, face_landmarks, right_cheek_indices, width, height)
                            roi_nose = rppg.landmark_extraction(frame, face_landmarks, nose_indices, width, height)
                            roi_ulip = rppg.landmark_extraction(frame, face_landmarks, upper_lip_indices, width, height)

                        right_cheek.append(roi_r)
                        nose.append(roi_nose)
                        ulip.append(roi_ulip)
                        
            if f>=int((W-SW)*CAMERA_FPS) and f%int(CAMERA_FPS)==0:
                cv2.imshow("Video Stream", disp_frames[f])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        efs = int(W*CAMERA_FPS)
        if len(face_frames)==efs and len(nose)==efs and len(right_cheek)==efs and len(ulip)==efs:
            window_length_ = int(CAMERA_FPS/2)
            poly_order_ = 5
            sigma_ = 2.5
            sigma_bpm = 5
            fs_ = int(CAMERA_FPS)
            f1, f2 = 0.5, 2.5
            fn = 0.1
            Q = 50
            f_ord = 10

            rc_ppgi = rppg.compute_landmark_ppgi(right_cheek)
            nose_ppgi = rppg.compute_landmark_ppgi(nose)
            face_ppgi = rppg.compute_landmark_ppgi(face_frames)
            ulip_ppgi = rppg.compute_landmark_ppgi(ulip)

            rc_ppg = rppg.ppgi_2_ppg(rc_ppgi,f1,f2,fs_,f_ord,fn,Q)
            nose_ppg = rppg.ppgi_2_ppg(nose_ppgi,f1,f2,fs_,f_ord,fn,Q)
            face_ppg = rppg.ppgi_2_ppg(face_ppgi,f1,f2,fs_,f_ord,fn,Q)
            ulip_ppg = rppg.ppgi_2_ppg(ulip_ppgi,f1,f2,fs_,f_ord,fn,Q)
            
            D = 3
            rc_ppg = np.abs(rc_ppg[int(D*CAMERA_FPS):])
            nose_ppg = np.abs(nose_ppg[int(D*CAMERA_FPS):])
            face_ppg = np.abs(face_ppg[int(D*CAMERA_FPS):])
            ulip_ppg = np.abs(ulip_ppg[int(D*CAMERA_FPS):])

            rc_ppg = rppg.savitzky_golay(rc_ppg, window_length_, poly_order_)
            nose_ppg = rppg.savitzky_golay(nose_ppg, window_length_, poly_order_)
            face_ppg = rppg.savitzky_golay(face_ppg, window_length_, poly_order_)
            ulip_ppg = rppg.savitzky_golay(ulip_ppg, window_length_, poly_order_)

            rc_ppg = rppg.gaussian_filter(rc_ppg, sigma_)
            nose_ppg = rppg.gaussian_filter(nose_ppg, sigma_)
            face_ppg = rppg.gaussian_filter(face_ppg, sigma_)
            ulip_ppg = rppg.gaussian_filter(ulip_ppg, sigma_)

            rc_ppg = rppg.min_max_sig(rc_ppg)
            nose_ppg = rppg.min_max_sig(nose_ppg)
            face_ppg = rppg.min_max_sig(face_ppg)
            ulip_ppg = rppg.min_max_sig(ulip_ppg)

            r_hr = round(hr_method(rc_ppg,fs_), 2)
            nose_hr = round(hr_method(nose_ppg,fs_), 2)
            face_hr = round(hr_method(face_ppg,fs_), 2)
            ulip_hr = round(hr_method(ulip_ppg,fs_), 2)
            
            # r_hr = round(hr_method(rc_ppg,fs_,sigma_bpm), 2)
            # nose_hr = round(hr_method(nose_ppg,fs_,sigma_bpm), 2)
            # face_hr = round(hr_method(face_ppg,fs_,sigma_bpm), 2)
            # ulip_hr = round(hr_method(ulip_ppg,fs_,sigma_bpm), 2)

            hr_avg = round((r_hr+nose_hr+face_hr+ulip_hr)/4, 2)
            HR_estimates = HR_estimates + [1.01*r_hr, 1.0*nose_hr, 1.0*face_hr, 1.01*ulip_hr]

            ppg = np.stack([nose_ppg, rc_ppg, face_ppg, ulip_ppg], axis=0)
            for i, line in enumerate(lines):
                line.set_data(np.linspace(0,W-D,int((W-D)*CAMERA_FPS)),ppg[i, :])  # Update x and y data       
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            if len(HR_estimates)>=1:
                print(len(HR_estimates), HR_estimates)
                HR_S = HR_estimates
                # HR_S = rppg.parzen_rosenblatt_window(HR_estimates,3)
                # HR_S = rppg.gaussian_kernel(HR_estimates)
                # HR_S = rppg.hr_gaussian_window(HR_estimates, 0, 10)
                print(len(HR_S), HR_S)
                HR_ = np.mean(HR_S)
                heart_rate = HR_
                print(f"HEART RATE: {HR_}bpm")
                print("----------------------------------------------------------------------------------------------------------------------------------------------------")
            if len(HR_estimates)==(int(W/SW))*len(channels):      
                HR_estimates = []
            # print("----------------------------------------------------------------------------------------------------------------------------------------------------")
        else:
            print("EFS not met")
            print(efs, len(face_frames), len(nose), len(right_cheek), len(ulip))
        if SW < W:
            frames = frames[int(SW*CAMERA_FPS):]
        else:
            frames = []

    # Release the capture when everythin is done
    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    record()