import cv2
import mediapipe as mp
import math
from fer import FER

# ----------------- Setup -----------------
# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_hand_draw = mp.solutions.drawing_utils

# OpenCV Haarcascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# FER Emotion Detector (no MoviePy needed)
detector = FER(mtcnn=True)

# Punch detection memory
prev_z = None
punch_detected = False

# ----------------- Video Capture -----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # mirror view
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = frame.copy()

    # ----------------- Hand Gesture Detection -----------------
    hand_results = hands.process(rgb_frame)
    gesture_text = "No Gesture"

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_hand_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Closed hand check (fist)
            tips = [8, 12, 16, 20]   # fingertips
            mcps = [5, 9, 13, 17]    # MCP joints
            closed_count = 0
            for t, m in zip(tips, mcps):
                tip_y = hand_landmarks.landmark[t].y
                mcp_y = hand_landmarks.landmark[m].y
                if tip_y > mcp_y:  # finger folded
                    closed_count += 1

            if closed_count >= 3:  # fist detected
                gesture_text = "Closed Hand (Fist)"
                # Punch detection using wrist z movement
                wrist_z = hand_landmarks.landmark[0].z
                if prev_z is not None and wrist_z < prev_z - 0.05:  # moving forward
                    punch_detected = True
                    gesture_text = "Punch Detected!"
                prev_z = wrist_z
            else:
                prev_z = None
                punch_detected = False

    # ----------------- Face Count -----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    face_count = len(faces)

    # ----------------- Emotion Detection -----------------
    emotions = detector.detect_emotions(frame)
    emotion_text = "Neutral"

    if emotions:
        best_face = emotions[0]  # take first detected face
        (x, y, w_box, h_box) = best_face["box"]

        # Get top emotion
        top_emotion, score = detector.top_emotion(frame)
        if top_emotion:
            emotion_text = f"{top_emotion} ({score:.2f})"

        # Draw bounding box + emotion
        cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(display_frame, emotion_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ----------------- Display Results -----------------
    cv2.putText(display_frame, f'Faces: {face_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(display_frame, f'Emotion: {emotion_text}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f'Gesture: {gesture_text}', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Facial Emotion + Face Count + Gesture", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
