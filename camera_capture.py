import cv2
import mediapipe as mp
import numpy as np
import serial
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- MediaPipe setup ----------------

ser = serial.Serial('/dev/cu.usbmodemB43A45B988EC2', 9600, timeout=1)
time.sleep(2)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 30)
indexOrg = (10, 70)
middleOrg = (10, 110)
ringOrg = (10, 150)
pinkyOrg = (10, 190)
color = (0, 0, 0)  
font_scale = 1
thickness = 4


calibration_mode = True
value_initialized = False
good = False

ideal_ratio = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]  


BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, 
    num_hands=2
)
detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

#  ---------------- Hand connection indices -----------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), 
    (0,5),(5,6),(6,7),(7,8), 
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ---------------- Hand landmark drawing function -----------------

def draw_landmarks(image, landmarks, r,g,b):
    h, w = image.shape[:2]
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (r, g, b), -1)
    
    for connection in HAND_CONNECTIONS:
        start = landmarks[connection[0]]
        end = landmarks[connection[1]]
        cv2.line(image, (int(start.x*w), int(start.y*h)), 
                (int(end.x*w), int(end.y*h)), (r, g, b), 2)
        
# ------------------- Initialize timestamp counter -------------------
timestamp_ms = 0

def get_hand_type(results):
    for i, hand_landmarks in enumerate(results.hand_landmarks):
        hand_type = results.handedness[i][0].category_name 
        if hand_type == "Right":
            return "Right"
        elif hand_type == "Left":
            return "Left"
        
#-------------------- Mapping values from ideal to 0-1 -------------------

def mapping(value, ideal):
    return((value-ideal)/(1-ideal))

# ------------------- Main loop -------------------

while cap.isOpened():
    #Setup MediaPipe Hands with custom parameters
    success, image = cap.read()
    if not success: continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  

    #-------------------- finger state detection --------------------

    if results.hand_landmarks:
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            hand_type = results.handedness[i][0].category_name
            landmarks = hand_landmarks

            # Thumb detection
            thumb_extended = 0
            if hand_type == "Right":
                right_thumb_extended = 1 if landmarks[4].x < landmarks[3].x else 0
            else: 
                left_thumb_extended = 1 if landmarks[4].x > landmarks[3].x else 0   

            mapped_finger_state_right = [0, 0, 0, 0, 0] 

            # Finger detection
            if hand_type == "Right":
                finger_state_right = [0, 0, 0, 0, 0]
                finger_state_right[0] = right_thumb_extended
            if hand_type == "Left":
                finger_state_left = [0, 0, 0, 0, 0]
                finger_state_left[0] = left_thumb_extended

            tips = [8, 12, 16, 20]
            bases = [5, 9, 13, 17]
            lower_tips = [6, 10, 14, 18]
            upper_tips = [7, 11, 15, 19]

            for i in range(4):
                if hand_type == "Left":
                    if landmarks[tips[i]].y < landmarks[bases[i]].y:
                        finger_state_left [i+1] = 0
                    else : 
                        finger_state_left[i+1] = 1
                else:    
                    if landmarks[tips[i]].y > landmarks[bases[i]].y:
                        finger_state_right [i+1] = 1
                    else : 
                        finger_state_right[i+1] = (landmarks[tips[i]].y/landmarks[bases[i]].y) 

            #-------------------- calibration ---------------------
            
            if calibration_mode == True:
                cv2.putText(image, f"Calibration Mode", org, 
                        font, font_scale, color, thickness)
                if(hand_type == "Right"):
                    draw_landmarks(image, hand_landmarks,255,0,0)
                key = cv2.waitKey(1)  # Wait for a key press to proceed with calibration
                if key == ord('c'):  # Press 'c' to calibrate
                    for i in range(4):
                        ideal_ratio[i][0] = landmarks[tips[i]].y/landmarks[bases[i]].y
                        ideal_ratio[i][1] = landmarks[lower_tips[i]].y/landmarks[bases[i]].y
                        ideal_ratio[i][2] = landmarks[upper_tips[i]].y/landmarks[bases[i]].y
                        value_initialized = True
                    calibration_mode = False
                



            #---------------------- run mode ----------------------
            if calibration_mode == False:  
                key = cv2.waitKey(1)
                if key == ord('r'):
                    calibration_mode = True

                if(hand_type == "Right"):
                    draw_landmarks(image, hand_landmarks,0,255,0)

               
                
                for i in range(4):
                    if abs((ideal_ratio[i][0] /(landmarks[tips[i]].y/landmarks[bases[i]].y)) - (ideal_ratio[i][2] /(landmarks[upper_tips[i]].y/landmarks[bases[i]].y))) < 0.008:
                        mapped_finger_state_right[i] = 0


                    if finger_state_right[i+1] != 0 and finger_state_right[i+1] != 1:
                        finger_state_right[i+1] = abs(mapping(finger_state_right[i+1], ideal_ratio[i][0]))

                if hand_type == "Right":
                    cv2.putText(image, f"good to go!", org, 
                    font, font_scale, color, thickness)
                    cv2.putText(image, f"Index Finger state: {finger_state_right[1]:.2f}", indexOrg, 
                    font, font_scale, color, thickness)
                    cv2.putText(image, f"Middle Finger state: {finger_state_right[2]:.2f}", middleOrg, 
                    font, font_scale, color, thickness)
                    cv2.putText(image, f"Ring Finger state: {finger_state_right[3]:.2f}", ringOrg, 
                    font, font_scale, color, thickness)
                    cv2.putText(image, f"Pinky Finger state: {finger_state_right[4]:.2f}", pinkyOrg, 
                    font, font_scale, color, thickness)
                    ser.write(f"{finger_state_right[0]:.2f},{finger_state_right[1]:.2f},{finger_state_right[2]:.2f},{finger_state_right[3]:.2f},{finger_state_right[4]:.2f}!\n".encode())
            
            
            

            
    
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == 27: break # Escape key to exit.

cap.release()
cv2.destroyAllWindows()
