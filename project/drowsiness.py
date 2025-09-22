import cv2
import numpy as np
import mediapipe as mp
import time
import winsound  # For beep sound on Windows

# Initialize MediaPipe Face Mesh and drawing utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks (RIGHT and LEFT eye indices in 468 points mesh)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Calculate Eye Aspect Ratio
def eye_aspect_ratio(landmarks, eye_indices):
    p2 = landmarks[eye_indices[1]]
    p6 = landmarks[eye_indices[5]]
    p3 = landmarks[eye_indices[2]]
    p5 = landmarks[eye_indices[4]]
    p1 = landmarks[eye_indices[0]]
    p4 = landmarks[eye_indices[3]]
    A = np.linalg.norm(np.array(p2) - np.array(p6))
    B = np.linalg.norm(np.array(p3) - np.array(p5))
    C = np.linalg.norm(np.array(p1) - np.array(p4))
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EAR_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 20
counter = 0
drowsy_start_time = None
drowsy_duration = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))

        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        avg_ear = (right_ear + left_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            counter += 1

            # Start timing when drowsiness is first detected
            if counter == EAR_CONSEC_FRAMES:
                drowsy_start_time = time.time()

            if counter >= EAR_CONSEC_FRAMES:
                drowsy_duration = int(time.time() - drowsy_start_time)
                cv2.putText(frame, f"DROWSINESS ALERT!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, f"Drowsy Time: {drowsy_duration} sec", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                winsound.Beep(1000, 500)  # Beep: 1000Hz for 500ms
        else:
            counter = 0
            drowsy_start_time = None
            drowsy_duration = 0

        # Draw eye landmarks
        for idx in RIGHT_EYE + LEFT_EYE:
            try:
                x, y = landmarks[idx]
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            except IndexError:
                pass

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
