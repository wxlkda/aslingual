import cv2
import time
import mediapipe as mp
import numpy as np

class HandGestureModel:
    def __init__(self):
        pass

    def predict(self, landmarks):
        thumb_is_open = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP].y < landmarks[mp.solutions.hands.HandLandmark.THUMB_MCP].y
        index_is_open = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y
        middle_is_open = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y
        ring_is_open = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_MCP].y
        pinky_is_open = landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP].y < landmarks[mp.solutions.hands.HandLandmark.PINKY_MCP].y
        
        fingers_open = np.array([thumb_is_open, index_is_open, middle_is_open, ring_is_open, pinky_is_open])
        num_fingers_open = np.sum(fingers_open)

        if num_fingers_open == 5:
            return "Open Hand"
        elif num_fingers_open == 0:
            return "Closed Fist"
        else:
            return "Unknown Gesture"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

WIDTH = 1366
HEIGHT = 1080
FONT_SCALE = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 2
FPS = 30

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

frame_counter = 0
start_time = time.time()

left_hand_detected = False
right_hand_detected = False

gesture_model = HandGestureModel()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    left_hand_detected = False
    right_hand_detected = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = gesture_model.predict(hand_landmarks.landmark)
            cv2.putText(frame, gesture, (10, 60), FONT, FONT_SCALE, (255, 255, 255), THICKNESS)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                left_hand_detected = True
            else:
                right_hand_detected = True

    if left_hand_detected:
        cv2.putText(frame, "Left Hand Detected", (10, frame.shape[0] - 10), FONT, FONT_SCALE, (0, 255, 0), THICKNESS)
    else:
        cv2.putText(frame, "Left Hand Not Detected", (10, frame.shape[0] - 10), FONT, FONT_SCALE, (0, 0, 255), THICKNESS)

    if right_hand_detected:
        cv2.putText(frame, "Right Hand Detected", (WIDTH - (cv2.getTextSize("Right Hand Detected", FONT, FONT_SCALE, THICKNESS)[0][0] * 2) + 10, frame.shape[0] - 10), FONT, FONT_SCALE, (0, 255, 0), THICKNESS)
    else:
        cv2.putText(frame, "Right Hand Not Detected", (WIDTH - (cv2.getTextSize("Right Hand Not Detected", FONT, FONT_SCALE, THICKNESS)[0][0] * 2) + 10, frame.shape[0] - 10), FONT, FONT_SCALE, (0, 0, 255), THICKNESS)
    
    frame_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        FPS = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {FPS:.2f}", (10, 30), FONT, FONT_SCALE, (0, 255, 0), THICKNESS)
    cv2.putText(frame, "Q to quit", (WIDTH - (cv2.getTextSize("Q to quit", FONT, FONT_SCALE, THICKNESS)[0][0] * 2), 30), FONT, FONT_SCALE, (0, 255, 0), THICKNESS)
    cv2.imshow('Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
