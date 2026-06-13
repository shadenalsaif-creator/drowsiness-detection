import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras

# ==================== الصوت ====================
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert_ar.mp3")
last_alert_time = 0
ALERT_COOLDOWN = 3

# ==================== النموذج المدرب (هيكل + أوزان) ====================
def build_model():
    return keras.Sequential([
        keras.layers.Input(shape=(64, 64, 1)),
        keras.layers.Rescaling(1./255),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid")
    ])

eye_model = build_model()
eye_model.load_weights("eye_weights.weights.h5")
print("النموذج المدرب جاهز!")

def classify_eye(gray, x, y, w, h):
    """يقص العين ويسأل الشبكة العصبية: مفتوحة ولا مغمضة؟"""
    crop = gray[y:y+h, x:x+w]
    crop = cv2.resize(crop, (64, 64))
    crop = crop.reshape(1, 64, 64, 1).astype("float32")
    prob = float(eye_model.predict(crop, verbose=0)[0][0])
    return prob  # قريب من 1 = مفتوحة، قريب من 0 = مغمضة

# ==================== الإعدادات ====================
CLOSED_TIME_LIMIT = 1.5      # ثواني الإغماض = نعاس (وجه مكشوف)
NIQAB_CLOSED_LIMIT = 2.0     # ثواني الإغماض = نعاس (نقاب)
FACE_LOST_LIMIT = 1.0        # ثواني بدون وجه = وضع العيون فقط

# ==================== mediapipe (تحديد مكان العين) ====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]  # نقاط منطقة الفم (لفحص التغطية)

# ==================== كاشف العيون (Haar) ====================
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance(gray):
    """تحسين تكيفي للإضاءة: يفتح الظلام ويخفف الشمس القوية"""
    brightness = gray.mean()
    if brightness < 80:
        gamma = 1.8
    elif brightness > 180:
        gamma = 0.6
    else:
        gamma = 1.0
    if gamma != 1.0:
        table = np.array([
            ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
        ]).astype("uint8")
        gray = cv2.LUT(gray, table)
    return clahe.apply(gray)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def eye_aspect_ratio(landmarks, eye_points):
    v1 = distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    v2 = distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    h = distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    return (v1 + v2) / (2.0 * h)

def eye_crop_classify(gray, landmarks, eye_points, w, h):
    """يقص منطقة العين من نقاط mediapipe ويسأل الشبكة العصبية"""
    xs = [landmarks[i].x for i in eye_points]
    ys = [landmarks[i].y for i in eye_points]
    x1, x2 = int(min(xs) * w), int(max(xs) * w)
    y1, y2 = int(min(ys) * h), int(max(ys) * h)
    mx, my = int((x2 - x1) * 0.4), int((y2 - y1) * 1.2)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, None
    return classify_eye(gray, x1, y1, x2 - x1, y2 - y1), (x1, y1, x2, y2)

def is_face_covered(gray, landmarks, w, h):
    """يفحص منطقة الفم: قماش موحد = وجه مغطى (نقاب/كمامة)"""
    xs = [landmarks[i].x for i in MOUTH]
    ys = [landmarks[i].y for i in MOUTH]
    x1, x2 = int(min(xs) * w) - 15, int(max(xs) * w) + 15
    y1, y2 = int(min(ys) * h) - 15, int(max(ys) * h) + 15
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return False
    patch = gray[y1:y2, x1:x2]
    return patch.std() < 18  # تباين منخفض = قماش موحد

def run_eyes_cnn(frame, gray, now):
    """مسار النقاب/العيون فقط: Haar يحدد + CNN يحكم"""
    global last_eyes_seen, ever_seen_eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05,
                                        minNeighbors=4, minSize=(25, 25))
    open_eye_found = False
    for (ex, ey, ew, eh) in eyes:
        prob = classify_eye(gray, ex, ey, ew, eh)
        is_open = prob > 0.5
        color = (0, 255, 0) if is_open else (0, 0, 255)
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
        cv2.putText(frame, f"{'OPEN' if is_open else 'CLOSED'} {prob:.2f}",
                    (ex, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if is_open:
            open_eye_found = True

    if open_eye_found:
        last_eyes_seen = now
        ever_seen_eyes = True
    if ever_seen_eyes and now - last_eyes_seen >= NIQAB_CLOSED_LIMIT:
        fire_alert(frame, now)

def fire_alert(frame, now):
    global last_alert_time, alerts_count
    cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    if now - last_alert_time > ALERT_COOLDOWN:
        alert_sound.play()
        last_alert_time = now
        alerts_count += 1

# ==================== الحالة ====================
cap = cv2.VideoCapture(0)
closed_start = None
last_face_time = time.time()
last_eyes_seen = time.time()
ever_seen_eyes = False
alerts_count = 0
session_start = time.time()

print("النظام شغال! اضغطي q للخروج")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    now = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = enhance(gray)
    h, w = frame.shape[:2]

    if results.multi_face_landmarks:
        last_face_time = now
        landmarks = results.multi_face_landmarks[0].landmark

        if is_face_covered(gray, landmarks, w, h):
            # ========== وجه مغطى (نقاب): نتجاهل نقاط mediapipe ==========
            mode = "NIQAB MODE (CNN)"
            run_eyes_cnn(frame, gray, now)
        else:
            # ========== وجه مكشوف: mediapipe يحدد + CNN يحكم ==========
            mode = "FACE MODE (CNN)"
            ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
                   eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0

            probs = []
            for eye_pts in [LEFT_EYE, RIGHT_EYE]:
                prob, box = eye_crop_classify(gray, landmarks, eye_pts, w, h)
                if prob is not None:
                    probs.append(prob)
                    x1, y1, x2, y2 = box
                    color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"EAR: {ear:.3f}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if probs:
                avg_prob = sum(probs) / len(probs)
                eyes_open = avg_prob > 0.5
                cv2.putText(frame,
                            f"CNN: {'OPEN' if eyes_open else 'CLOSED'} {avg_prob:.2f}",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if eyes_open else (0, 0, 255), 2)

                if not eyes_open:
                    if closed_start is None:
                        closed_start = now
                    if now - closed_start >= CLOSED_TIME_LIMIT:
                        fire_alert(frame, now)
                else:
                    closed_start = None

    elif now - last_face_time > FACE_LOST_LIMIT:
        # ========== ما فيه وجه أصلاً: وضع العيون فقط ==========
        mode = "EYES-ONLY MODE (CNN)"
        run_eyes_cnn(frame, gray, now)
    else:
        mode = "SEARCHING..."

    # ==================== معلومات الشاشة ====================
    minutes = int((now - session_start) // 60)
    cv2.putText(frame, f"Mode: {mode}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    cv2.putText(frame, f"Alerts: {alerts_count} | Session: {minutes} min",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Drowsiness Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
