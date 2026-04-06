import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- MediaPipe setup ----------------

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

ideal_ratio = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]  

good = False

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
# ------------------get hand----------------
def get_hand(): 
        print(results.handedness)
        # for i in enumerate(results.hand_landmarks):
        #     hand_type = results.handedness[i][0].category_name
        # return hand_type

#  ---------------- Hand connection indices -----------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), 
    (0,5),(5,6),(6,7),(7,8), 
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


# ---------------- Saving ideal ratios function -----------------
def save():
    hand = results.handedness[0][0].category_name
    if(hand == "Right"):
        for i in range(4):
            if hand== "Right":
                ideal_ratio[i][0] = landmarks[tips[i]].y/landmarks[bases[i]].y
                ideal_ratio[i][1] = landmarks[lower_tips[i]].y/landmarks[bases[i]].y
                ideal_ratio[i][2] = landmarks[upper_tips[i]].y/landmarks[bases[i]].y
                print("values savwd")

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

# def get_hand_type(hand_landmarks, image_width, image_height):
#     # Normalize landmarks to image coordinates
#     thumb_tip = hand_landmarks[4]  # Thumb tip (landmark 4)
#     wrist = hand_landmarks[0]      # Wrist (landmark 0)
    
#     thumb_x = thumb_tip.x * image_width
#     wrist_x = wrist.x * image_width
    
#     # Right hand: thumb on LEFT side of wrist (mirrored view)
#     # Left hand: thumb on RIGHT side of wrist
#     return "Right" if thumb_x < wrist_x else "Left"

# ------------------- Main loop -------------------

while cap.isOpened():
    #Setup MediaPipe Hands with custom parameters
    success, image = cap.read()
    if not success: continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  

   # ------------------- Drawing only Right -------------------

    if results.hand_landmarks:
        
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            hand_type = results.handedness[i][0].category_name 
            if hand_type == "Right":
                draw_landmarks(image, hand_landmarks,0,255,0)

    

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
                if hand_type=="Right":
                    cv2.putText(image, f"First Ratio {landmarks[tips[i]].y/landmarks[bases[i]].y:.2f}", indexOrg, 
                            font, font_scale, color, thickness)
                
                if(hand_type == "Right"):
                    draw_landmarks(image, hand_landmarks,255,0,0)

                if(hand_type == "Left"):
                    draw_landmarks(image, hand_landmarks,0,0,255)
                    print(finger_state_left)
                    if(finger_state_left[0] == 1 and finger_state_left[1] == 1 and finger_state_left[2] == 1 and finger_state_left[3] == 1 and finger_state_left[4] == 1):
                        save()  
                        calibration_mode = False
                        print(ideal_ratio)
                        print("Right Hand : " + str(hand_type == "Right"))
                        print("Left Hand : "+ str(hand_type == "Left"))
                        print("Calibration complete. Starting main mode.")


            #---------------------- run mode ----------------------
            if calibration_mode == False:
                for i in range(4):
                    if(landmarks[tips[i]].y/landmarks[bases[i]].y == ideal_ratio[i][0]):
                        finger_state_right[i+1] = 0
                    if(landmarks[lower_tips[i]].y/landmarks[bases[i]].y == ideal_ratio[i][1]):
                        finger_state_right[i+1] = 0
                    if(landmarks[upper_tips[i]].y/landmarks[bases[i]].y == ideal_ratio[i][2]):
                        finger_state_right[i+1] = 0
                if (landmarks[tips[0]].y <landmarks[tips[1]].y or landmarks[tips[2]].y < landmarks[tips[1]].y) and finger_state_right[1] < 0.5 and finger_state_right[2] < 0.5 and finger_state_right[3] < 0.5 and finger_state_right[4] < 0.5:
                    if hand_type == "Right":
                        cv2.putText(image, f"please Straighten your hand", org, 
                        font, font_scale, color, thickness)
                else :
                    if hand_type == "Right":
                        cv2.putText(image, f"good to go!", org, 
                        font, font_scale, color, thickness)
                        cv2.putText(image, f"Index Finger Y: {ideal_ratio[0][0]:.2f}", indexOrg, 
                        font, font_scale, color, thickness)
                        cv2.putText(image, f"Index Finger X: {landmarks[tips[i]].y/landmarks[bases[i]].y:.2f}", middleOrg, 
                        font, font_scale, color, thickness)

            
            

            
    
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == 27: break # Escape key to exit.

cap.release()
cv2.destroyAllWindows()
