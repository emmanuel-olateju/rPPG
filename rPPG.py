import yaml
import numpy as np
import cv2
import torch
from modules import processing, neural_nets

model = neural_nets.rPPG_PSD_MLP()
model.load_state_dict(torch.load("artifacts/model.sav"))
model.eval()

with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

frame_standardization = processing.StandardizeFrame(
    target_resolution=CONFIG["TARGET_RESOLUTION"],
    target_frame_rate=CONFIG["TARGET_FRAME_RATE"],
    target_color_space=cv2.COLOR_BGR2RGB)

face_detect = processing.FaceDetection()
ppgi = processing.PPGIcomputation(alpha=0.1,filter_size=CONFIG["MEDIAN_FILTER_SIZE"],target_frame_size=CONFIG["TARGET_RESOLUTION"])
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

    # Initialize Video Capture
    cap = cv2.VideoCapture(CONFIG["VIDEO"])
    CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CAMERA_FPS = cap.get(cv2.CAP_PROP_FPS)
    print((CAMERA_WIDTH,CAMERA_HEIGHT), CAMERA_FPS)

    # Check if camers opened succesfully
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return


    # Initialize Video Capture
    cap = cv2.VideoCapture(0)
    CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CAMERA_FPS = cap.get(cv2.CAP_PROP_FPS)
    print((CAMERA_WIDTH,CAMERA_HEIGHT), CAMERA_FPS)

    frames = []
    heart_rate = 0
    heart_rates = []
    while True:
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        ''' DETECT FACE & SHOW FACE '''
        faces, frame = face_detect.extract([frame])
        frame = frame[0].astype("uint8")
        face = faces[0]
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            text_x = x
            text_y = y + h + 25
            cv2.putText(frame, f"{heart_rate} BPM", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Video Stream", frame)

        if len(faces)==1:
            for (x, y, w, h) in face:
                frame = frame[y+5:(y+h)-5,x+5:(x+w)-5,:]
                frames.append(frame)
        else:
            frames = []

        # Break the loop on "q" key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if len(frames)==int(3*CAMERA_FPS):

            ''' STANDARDIZE FRAME '''
            frames = frame_standardization.standardize(frames,current_frame_rate=CAMERA_FPS)
            frames = [((frame-frame.min())/(frame.max()-frame.min()))*255 for frame in frames]
            frame = [frame.astype("uint8") for frame in frames]

            ''' EXTRACT PPG SIGNAL '''
            skin_frames, cks = ppgi.extract_face(frames)
            ppgi_signal = ppgi.compute_ppgi(skin_frames,cks)
            ppg_signal = ppgi.compute_ppg(np.array(ppgi_signal))
            ppg_signal = processing.butter_bandpass_filter(ppg_signal, 0.4, 4, 30, order=4)
            ppg_signal = (ppg_signal-ppg_signal.mean())/ppg_signal.std()

            ''' COMPUTE PSD & HR'''
            # frequencies, psd = HR.get_psd(ppg_signal)
            # psd = (psd-psd.min())/(psd.max()-psd.min())
            # psd = np.vstack((frequencies, psd))
            # psd = np.expand_dims(psd, axis=0).astype(np.float32)
            # psd = torch.tensor(psd)
            # est_psd = model(psd).detach().numpy()
            # est_psd = np.squeeze(est_psd,axis=0)
            # hr = round(HR.compute_from_psd(frequencies, est_psd), ndigits=None)
            frames = []
            hr_ = round(HR.compute(ppg_signal), ndigits=2)
            # heart_rate = round(0.3*hr + 0.7*hr_, ndigits=None)
            heart_rates.append(hr_)
            if len(heart_rates) > 5:
                _ = heart_rates.pop(0)
            heart_rate = round(sum(heart_rates)/len(heart_rates), ndigits=2)
            print(f"HEART RATE: {heart_rate}bpm | PSD:HR: {hr_}bpm")

    
    # Release the capture when everythin is done
    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    record()