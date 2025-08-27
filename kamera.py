import cv2
import mediapipe as mp
from deepface import DeepFace
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_hand_draw = mp.solutions.drawing_utils

# OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Helper function for distance
def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

# Start Video Capture
cap = cv2.VideoCapture(0)
frame_counter = 0
gender = "Unknown"
emotion_text = "Neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = frame.copy()

    # ---- Hand Gesture Detection ----
    hand_results = hands.process(rgb_frame)
    gesture_text = "No Gesture"
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_hand_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_text = "Hand Detected"

    # ---- Face Detection for Gender ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w_box, h_box) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)
        face_img = frame[y:y+h_box, x:x+w_box]
        frame_counter += 1
        if frame_counter % 5 == 0:
            try:
                face_resized = cv2.resize(face_img, (224, 224))
                analysis = DeepFace.analyze(face_resized, actions=['gender'], enforce_detection=False)
                gender = analysis['gender']
            except:
                gender = "Unknown"
        break  # Only first face

    # ---- Face Mesh for Emotion ----
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=display_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # Extract landmarks for simple emotion estimation
            left_mouth = (int(face_landmarks.landmark[61].x * w), int(face_landmarks.landmark[61].y * h))
            right_mouth = (int(face_landmarks.landmark[291].x * w), int(face_landmarks.landmark[291].y * h))
            top_lip = (int(face_landmarks.landmark[13].x * w), int(face_landmarks.landmark[13].y * h))
            bottom_lip = (int(face_landmarks.landmark[14].x * w), int(face_landmarks.landmark[14].y * h))
            left_eyebrow = (int(face_landmarks.landmark[105].x * w), int(face_landmarks.landmark[105].y * h))
            right_eyebrow = (int(face_landmarks.landmark[334].x * w), int(face_landmarks.landmark[334].y * h))
            left_eye_top = (int(face_landmarks.landmark[159].x * w), int(face_landmarks.landmark[159].y * h))
            left_eye_bottom = (int(face_landmarks.landmark[145].x * w), int(face_landmarks.landmark[145].y * h))

            # Mouth ratio for smile/surprise
            mouth_width = distance(left_mouth, right_mouth)
            mouth_height = distance(top_lip, bottom_lip)
            if mouth_height / mouth_width > 0.3:
                emotion_text = "Surprised"
            elif mouth_height / mouth_width > 0.2:
                emotion_text = "Happy"
            else:
                emotion_text = "Neutral"

            # Eyebrow ratio for anger
            eyebrow_dist = distance(left_eyebrow, right_eyebrow)
            eye_height = distance(left_eye_top, left_eye_bottom)
            if eyebrow_dist / eye_height < 3.0:
                emotion_text = "Angry"

    # ---- Display results on camera ----
    cv2.putText(display_frame, f'Gender: {gender}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(display_frame, f'Emotion: {emotion_text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(display_frame, f'Gesture: {gesture_text}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Facial Emotion + Gender + Gesture", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()