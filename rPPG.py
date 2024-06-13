import yaml
import numpy as np
import cv2
import mediapipe as mp
from modules import processing


with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
left_cheek_indices = [50, 101, 118, 202, 203, 204, 205, 206, 207, 208, 209, 210]
right_cheek_indices = [280, 351, 437, 418, 417, 416, 415, 414, 413, 412, 411, 410]

frame_standardization = processing.StandardizeFrame(
    target_resolution=CONFIG["TARGET_RESOLUTION"],
    target_frame_rate=CONFIG["TARGET_FRAME_RATE"],
    target_color_space=cv2.COLOR_BGR2RGB)

face_detect = processing.FaceDetection()
ppgi = processing.PPGIcomputation(alpha=0.5,filter_size=CONFIG["MEDIAN_FILTER_SIZE"],target_frame_size=CONFIG["TARGET_RESOLUTION"])
HR = processing.HRcompute(
    CONFIG["BPF_LOWCUT"],
    CONFIG["BPF_HIGHCUT"],
    CONFIG["BPF_ORDER"],
    CONFIG["TARGET_FRAME_RATE"]
)
ibi_HR = processing.ibi_HRcompute(
    CONFIG["BPF_LOWCUT"],
    CONFIG["BPF_HIGHCUT"],
    CONFIG["BPF_ORDER"],
    CONFIG["TARGET_FRAME_RATE"]
)

def record():


    ''' Initialize Video Capture '''
    cap = cv2.VideoCapture(0)
    # Check if camers opened succesfully
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CAMERA_FPS = cap.get(cv2.CAP_PROP_FPS)
    print((CAMERA_WIDTH,CAMERA_HEIGHT), CAMERA_FPS)

    frames = []
    rois = []
    heart_rate = 0
    heart_rates = []
    while True:
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = (frame-frame.min())/(frame.max()-frame.min())*255
        frame = frame.astype("uint8")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ''' DETECT FACE & LANDMARKS '''
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                # Draw a rectangle around the detected face
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * CAMERA_WIDTH)
                y_min = int(bboxC.ymin * CAMERA_HEIGHT)
                box_width = int(bboxC.width * CAMERA_WIDTH)
                box_height = int(bboxC.height * CAMERA_HEIGHT)
                face_frame = frame[y_min+5:(y_min+box_height)-5,x_min+5:(x_min+box_width)-5,:]
                frames.append(face_frame)
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), (255, 255, 255), 2)
                text_x = x_min
                text_y = y_min + box_height + 25
                cv2.putText(frame, f"{heart_rate} BPM", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    roi_l = []
                    # Draw landmarks for the left cheek
                    for idx in left_cheek_indices:
                        x = int(face_landmarks.landmark[idx].x * int(0.9*CAMERA_WIDTH))
                        y = int(face_landmarks.landmark[idx].y * int(0.9*CAMERA_HEIGHT))
                        roi = frame[x,y,:]
                        roi_l.append(roi)
                    roi_l = np.array(roi_l)
                    roi_l = np.mean(roi_l,axis=0)
                    
                    roi_r = []
                    # Draw landmarks for the right cheek
                    for idx in right_cheek_indices:
                        x = int(face_landmarks.landmark[idx].x * int(0.9*CAMERA_WIDTH))
                        y = int(face_landmarks.landmark[idx].y * int(0.9*CAMERA_HEIGHT))
                        roi = frame[x,y,:]
                        roi_r.append(roi)
                    roi_r = np.array(roi_r)
                    roi_r = np.mean(roi_r,axis=0)
                rois.append((roi_r+roi_l)/2)

        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if len(frames)==int(4*CAMERA_FPS):

            rois = np.array(rois)

            ''' STANDARDIZE FRAME '''
            frames = frame_standardization.standardize(frames,current_frame_rate=CAMERA_FPS)
            frames = [((frame-frame.min())/(frame.max()-frame.min()))*255 for frame in frames]
            frame = [frame.astype("uint8") for frame in frames]

            ''' EXTRACT PPG SIGNAL '''
            # skin_frames, cks = ppgi.extract_face(frames)
            ppgi_signal = ppgi.compute_ppgi(frames,[0]*len(frames))
            if len(frames) == len(rois):
                ppg_signal_1 = ppgi.compute_ppg(np.array(ppgi_signal))
                ppg_signal_2 = ppgi.compute_ppg(rois)
                ppg_signal = 0.3*ppg_signal_1 + 0.7*ppg_signal_2
            else:
                ppg_signal = ppgi.compute_ppg(np.array(ppgi_signal))
            ppg_signal = processing.butter_bandpass_filter(ppg_signal, 0.4, 4, 30, order=4)
            ppg_signal = (ppg_signal-ppg_signal.mean())/ppg_signal.std()

            ''' COMPUTE PSD & HR'''
            frames = []
            hr = round(HR.compute(ppg_signal), ndigits=2)
            heart_rates.append(hr)
            heart_rate = round(sum(heart_rates[-2:])/len(heart_rates[-2:]), ndigits=2)
            if len(heart_rates) > 3:
                _ = heart_rates.pop(0)
            print(f"HEART RATE: {heart_rate}bpm | PSD:HR: {hr}bpm | {rois.shape} | {np.array(ppgi_signal).shape}")
            rois = []

    
    # Release the capture when everythin is done
    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    record()