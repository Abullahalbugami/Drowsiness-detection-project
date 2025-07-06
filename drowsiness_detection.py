import cv2
import mediapipe as mp
import time
from playsound import playsound

# إعداد شبكة الوجه باستخدام face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # تحويل الصورة من BGR إلى RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    # إذا تعرف على الوجه 
    if result.multi_face_landmarks:
     for face_landmarks in result.multi_face_landmarks:
        h, w = frame.shape[:2]

        RIGHT_EYE = [33, 133]
        LEFT_EYE = [362, 263]

        for i in RIGHT_EYE + LEFT_EYE:
            x = int(face_landmarks.landmark[i].x * w)
            y = int(face_landmarks.landmark[i].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    # عرض الصورة في نافذة
    cv2.imshow("Drowsiness Detection", frame)

    #تقفيل الكام بزر q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 
cap.release()
cv2.destroyAllWindows()
