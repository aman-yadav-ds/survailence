import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
buffer_len = 64
landmark_buffer = deque(maxlen=buffer_len)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    if results.multi_hand_landmarks:
        cv2.putText(frame, f'Hands detected: {len(results.multi_hand_landmarks)}', 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw connections & landmarks for each hand
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        # Build feature vector
        feat = np.zeros((2, 21, 3), dtype=np.float32)
        for i, lm in enumerate(results.multi_hand_landmarks):
            for p, l in enumerate(lm.landmark):
                feat[i, p] = [l.x, l.y, l.z]

        feat = feat.flatten()

    else:
        feat = np.zeros(2*21*3, dtype=np.float32)

    cv2.putText(frame, f'Buffer size: {len(landmark_buffer)}/{buffer_len}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    
    landmark_buffer.append(feat)

    cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
