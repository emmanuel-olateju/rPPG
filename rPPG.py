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

channels = ["nose_ppg", "rc_ppg", "lc_ppg", "ulip_ppg"]

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

        ''' DETECT FACE & LANDMARKS '''
        right_cheek = []
        left_cheek = []
        nose = []
        ulip = []
        llip = []
        disp_frames = copy.deepcopy(frames)
        for f,frame in enumerate(frames):

            frame_ = cv2.GaussianBlur(frame, (5,5), 0)
            height, width, _ = frame_.shape
            rgb_frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
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
                        cv2.rectangle(disp_frames[f], (x_min, y_min), (x_min + box_width, y_min + box_height), (255, 255, 255), 2)
                        text_x = x_min
                        text_y = y_min + box_height + 25
                        cv2.putText(disp_frames[f], f"{heart_rate:.2f} BPM", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            roi_r = rppg.landmark_extraction(frame_, face_landmarks, right_cheek_indices, width, height)
                            roi_l = rppg.landmark_extraction(frame_, face_landmarks, left_cheek_indices, width, height)
                            roi_nose = rppg.landmark_extraction(frame_, face_landmarks, nose_indices, width, height)
                            roi_ulip = rppg.landmark_extraction(frame_, face_landmarks, upper_lip_indices, width, height)
                            roi_llip = rppg.landmark_extraction(frame_, face_landmarks, lower_lip_indices, width, height)

                        right_cheek.append(roi_r)
                        left_cheek.append(roi_l)
                        nose.append(roi_nose)
                        ulip.append(roi_ulip)
                        llip.append(roi_llip)
                        
            if f>=int((W-SW)*CAMERA_FPS) and f%2==0:
                cv2.imshow("Video Stream", disp_frames[f])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        efs = int(W*CAMERA_FPS)
        if len(right_cheek)==efs and len(nose)==efs and len(ulip)==efs:
            window_length_ = int(CAMERA_FPS/2)
            poly_order_ = 5
            sigma_ = 2.5
            fs_ = int(CAMERA_FPS)
            f1, f2 = 0.5, 2.5
            fn = 0.1
            Q = 50
            f_ord = 10
            
            rc_Quality = rppg.area_measure_s(right_cheek)
            lc_Quality = rppg.area_measure_s(left_cheek)
            n_Quality = rppg.area_measure_s(nose)
            ul_Quality = rppg.area_measure_s(ulip)
            ll_Quality = rppg.area_measure_s(llip)
            gain = np.array([rc_Quality,lc_Quality, n_Quality, ul_Quality, ll_Quality])

            print(f"NOISE:{gain}")

            rc_ppgi = rppg.compute_landmark_ppgi(right_cheek)
            lc_ppgi = rppg.compute_landmark_ppgi(left_cheek)
            nose_ppgi = rppg.compute_landmark_ppgi(nose)
            ulip_ppgi = rppg.compute_landmark_ppgi(ulip)
            llip_ppgi = rppg.compute_landmark_ppgi(llip)

            rc_ppg = rppg.ppgi_2_ppg(rc_ppgi,f1,f2,fs_,f_ord,fn,Q)
            lc_ppg = rppg.ppgi_2_ppg(lc_ppgi,f1,f2,fs_,f_ord,fn,Q)
            nose_ppg = rppg.ppgi_2_ppg(nose_ppgi,f1,f2,fs_,f_ord,fn,Q)
            ulip_ppg = rppg.ppgi_2_ppg(ulip_ppgi,f1,f2,fs_,f_ord,fn,Q)
            llip_ppg = rppg.ppgi_2_ppg(llip_ppgi,f1,f2,fs_,f_ord,fn,Q)
            
            D = 2
            rc_ppg = np.abs(rc_ppg[int(D*CAMERA_FPS):])
            lc_ppg = np.abs(lc_ppg[int(D*CAMERA_FPS):])
            nose_ppg = np.abs(nose_ppg[int(D*CAMERA_FPS):])
            ulip_ppg = np.abs(ulip_ppg[int(D*CAMERA_FPS):])
            llip_ppg = np.abs(llip_ppg[int(D*CAMERA_FPS):])

            rc_ppg = rppg.savitzky_golay(rc_ppg, window_length_, poly_order_)
            lc_ppg = rppg.savitzky_golay(lc_ppg, window_length_, poly_order_)
            nose_ppg = rppg.savitzky_golay(nose_ppg, window_length_, poly_order_)
            ulip_ppg = rppg.savitzky_golay(ulip_ppg, window_length_, poly_order_)
            llip_ppg = rppg.savitzky_golay(llip_ppg, window_length_, poly_order_)

            rc_ppg = rppg.gaussian_filter(rc_ppg, sigma_)
            lc_ppg = rppg.gaussian_filter(lc_ppg, sigma_)
            nose_ppg = rppg.gaussian_filter(nose_ppg, sigma_)
            ulip_ppg = rppg.gaussian_filter(ulip_ppg, sigma_)
            llip_ppg = rppg.gaussian_filter(llip_ppg, sigma_)

            rc_hr = round(hr_method(rc_ppg,fs_), 2)
            lc_hr = round(hr_method(lc_ppg,fs_), 2)
            nose_hr = round(hr_method(nose_ppg,fs_), 2)
            ulip_hr = round(hr_method(ulip_ppg,fs_), 2)
            llip_hr = round(hr_method(llip_ppg,fs_), 2)

            areas_hr = np.array([rc_hr, lc_hr, nose_hr, ulip_hr, llip_hr])

            HR_estimates = HR_estimates + [areas_hr[np.argmax(gain)]]

            if len(HR_estimates)>=1:
                # print(len(HR_estimates), HR_estimates)
                HR_S = HR_estimates
                # HR_S = rppg.parzen_rosenblatt_window(HR_estimates,3)
                print(len(HR_S), HR_S)
                HR_ = np.mean(HR_S)
                heart_rate = HR_
                print(f"HEART RATE: {HR_}bpm | {np.argmax(gain)}")
                print("-------------------------------------------------------------------------------------------------------------------------------")
            if len(HR_estimates)>=len(channels):      
                HR_estimates = []
        else:
            print("EFS not met")
            print(efs, len(nose), len(right_cheek), len(ulip))
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