import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- MediaPipe setup ----------------

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

def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
    for connection in HAND_CONNECTIONS:
        start = landmarks[connection[0]]
        end = landmarks[connection[1]]
        cv2.line(image, (int(start.x*w), int(start.y*h)), 
                (int(end.x*w), int(end.y*h)), (0, 255, 0), 2)

# ------------------- Initialize timestamp counter -------------------
timestamp_ms = 0

# ------------------- Main loop -------------------

while cap.isOpened():
    #Setup MediaPipe Hands with custom parameters
    success, image = cap.read()
    if not success: continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            draw_landmarks(image, hand_landmarks)

    #finger state detection
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            landmarks = hand_landmarks

            # Thumb detection
            thumb_extended = 0
            hand_type = ""

            if hand_type == "Right":
                if landmarks[4].x < landmarks[3].x:
                    thumb_extended = 1
            else:  # Left hand
                if landmarks[4].x > landmarks[3].x:
                    thumb_extended = 1

            # Finger detection
            finger_state = [0, 0, 0, 0, 0]
            finger_state[0] = thumb_extended

            tips = [8, 12, 16, 20]
            bases = [5, 9, 13, 17]

            for i in range(4):
                if landmarks[tips[i]].y < landmarks[bases[i]].y:
                    finger_state[i+1] = 1
            

            
    
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == 27: break # Escape key to exit.

cap.release()
cv2.destroyAllWindows()
