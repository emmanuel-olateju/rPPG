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


with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

model = joblib.load("artifacts/WHR2HR_RandomForestRegressor.sav")
scaler = joblib.load("artifacts/HR_scaler.sav")

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

W = 15

def record():


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

    ppgi = processing.PPGIcomputation(alpha=0.5,filter_size=CONFIG["MEDIAN_FILTER_SIZE"],target_frame_size=CONFIG["TARGET_RESOLUTION"])
    HR = processing.HRcompute(
        CONFIG["BPF_LOWCUT"],
        CONFIG["BPF_HIGHCUT"],
        CONFIG["BPF_ORDER"],
        CONFIG["TARGET_FRAME_RATE"]
    )
    # ibi_HR = processing.ibi_HRcompute(
    #     CONFIG["BPF_LOWCUT"],
    #     CONFIG["BPF_HIGHCUT"],
    #     CONFIG["BPF_ORDER"],
    #     CONFIG["TARGET_FRAME_RATE"]
    # )

    frames = []
    heart_rate = 0
    while True:
        
        ''' 1S FRAME DATA ACQUISITION '''
        frames = []
        for i in range(int(W*CAMERA_FPS)):
            ret, frame = cap.read()
            if not ret:
                frames = []
                break
            else:
                frames.append(frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        ''' STANDARDIZE FRAMES '''
        # frames = [(frame-frame.min())/(frame.max()-frame.min())*255 for frame in frames]
        # frames = [frame.astype("uint8") for frame in frames]
        height, width = CAMERA_HEIGHT, CAMERA_WIDTH

        ''' DETECT FACE & LANDMARKS '''
        face_frames = []
        left_cheek = []
        right_cheek = []
        for frame in frames:

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
                        cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), (255, 255, 255), 2)
                        text_x = x_min
                        text_y = y_min + box_height + 25
                        cv2.putText(frame, f"{heart_rate:.2f} BPM", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            

                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:

                            roi_l = rppg.landmark_extraction(frame, face_landmarks, left_cheek_indices, width, height)
                            
                            roi_r = rppg.landmark_extraction(frame, face_landmarks, right_cheek_indices, width, height)

                        left_cheek.append(roi_l)
                        right_cheek.append(roi_r)

            cv2.imshow("Video Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        efs = int(W*CAMERA_FPS)
        if len(face_frames)==efs and len(left_cheek)==efs and len(right_cheek)==efs:

            cv2.imshow("Video Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            window_length_ = 10
            poly_order_ = 5
            sigma_ = 1
            fs_ = CAMERA_FPS
            f1, f2 = 0.75, 6
            fn = 0.1
            Q = 10000

            lc_ppgi = rppg.compute_landmark_ppgi(left_cheek)
            rc_ppgi = rppg.compute_landmark_ppgi(right_cheek)

            lc_ppg = rppg.ppgi_2_ppg(lc_ppgi,f1,f2,fs_,4,fn,Q)
            rc_ppg = rppg.ppgi_2_ppg(rc_ppgi,f1,f2,fs_,4,fn,Q)

            lc_ppg = rppg.savitzky_golay(lc_ppg, window_length_, poly_order_)
            rc_ppg = rppg.savitzky_golay(rc_ppg, window_length_, poly_order_)

            lc_ppg = rppg.gaussian_filter(lc_ppg, sigma_)
            rc_ppg = rppg.gaussian_filter(rc_ppg, sigma_)

            convG_ppg = np.convolve(lc_ppg, rc_ppg, mode="same")

            l_hr = round(rppg.median_analysis(lc_ppg,fs_),2)
            r_hr = round(rppg.median_analysis(rc_ppg,fs_),2)
            convG_hr = round(rppg.median_analysis(convG_ppg,fs_),2)


            heart_rate = round(0.7*r_hr + 0.3*l_hr, 2)
            print(f"HEART RATE: {heart_rate}bpm | r_hr: {r_hr}bpm | l_hr: {l_hr}bpm | convG_hr: {convG_hr}bpm")
            print("----------------------------------------------------------------------------------------------")
        else:
            frames = []

    # Release the capture when everythin is done
    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    record()