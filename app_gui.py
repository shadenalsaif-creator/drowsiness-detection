"""Driver Drowsiness Detection — Desktop GUI.

A dark-themed desktop app wrapping the detection engine:
  * Start / Stop monitoring
  * Custom Arabic alert phrase (regenerates alert_ar.mp3)
  * Live camera feed inside the window
  * Live mode / alert-count / session-timer dashboard

Run:
    python app_gui.py
"""

import os
import time
import math
import threading

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
import mediapipe as mp
import pygame
import tensorflow as tf
import keras
from gtts import gTTS

# ==================== المظهر ====================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ==================== النموذج المدرب ====================
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

def classify_eye(gray, x, y, w, h):
    crop = gray[y:y+h, x:x+w]
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (64, 64)).reshape(1, 64, 64, 1).astype("float32")
    return float(eye_model.predict(crop, verbose=0)[0][0])

# ==================== الإعدادات ====================
CLOSED_TIME_LIMIT = 1.5
NIQAB_CLOSED_LIMIT = 2.0
FACE_LOST_LIMIT = 1.0
ALERT_COOLDOWN = 3

# ==================== mediapipe ====================
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance(gray):
    b = gray.mean()
    gamma = 1.8 if b < 80 else (0.6 if b > 180 else 1.0)
    if gamma != 1.0:
        table = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
    return clahe.apply(gray)

def eye_crop_classify(gray, lm, pts, w, h):
    xs = [lm[i].x for i in pts]; ys = [lm[i].y for i in pts]
    cx, cy = int(sum(xs)/len(xs)*w), int(sum(ys)/len(ys)*h)
    ew = int((max(xs)-min(xs))*w*1.8); eh = int(ew*0.75)
    x1, y1 = max(0, cx-ew//2), max(0, cy-eh//2)
    x2, y2 = min(w, cx+ew//2), min(h, cy+eh//2)
    if x2-x1 < 10 or y2-y1 < 10:
        return None, None
    return classify_eye(gray, x1, y1, x2-x1, y2-y1), (x1, y1, x2, y2)

def is_face_covered(gray, lm, w, h):
    xs = [lm[i].x for i in MOUTH]; ys = [lm[i].y for i in MOUTH]
    x1, x2 = int(min(xs)*w)-15, int(max(xs)*w)+15
    y1, y2 = int(min(ys)*h)-15, int(max(ys)*h)+15
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
    if x2-x1 < 10 or y2-y1 < 10:
        return False
    return gray[y1:y2, x1:x2].std() < 18


class DrowsinessApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Driver Drowsiness Detection System")
        self.geometry("1100x640")

        pygame.mixer.init()
        self.alert_sound = None
        if os.path.exists("alert_ar.mp3"):
            self.alert_sound = pygame.mixer.Sound("alert_ar.mp3")

        self.running = False
        self.cap = None
        self.face_mesh = None
        self.reset_state()

        # ---- التخطيط: لوحة تحكم يسار + كاميرا يمين ----
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        panel = ctk.CTkFrame(self, width=300, corner_radius=15)
        panel.grid(row=0, column=0, padx=20, pady=20, sticky="ns")

        ctk.CTkLabel(panel, text="Drowsiness Detection",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(25, 5), padx=20)
        ctk.CTkLabel(panel, text= " النعاس كشف",
                     font=ctk.CTkFont(size=14), text_color="gray").pack(pady=(0, 20))

        ctk.CTkLabel(panel, text="Arabic alert phrase:\n " \
        "😍you can customize the alert sentence as you like😍",
                     anchor="w").pack(fill="x", padx=20)
        self.alert_entry = ctk.CTkEntry(panel, placeholder_text="انتبه! استيقظ")
        self.alert_entry.pack(fill="x", padx=20, pady=(5, 5))
        ctk.CTkButton(panel, text="Set alert voice",
                      command=self.set_alert).pack(fill="x", padx=20, pady=(0, 20))

        self.start_btn = ctk.CTkButton(panel, text="▶  Start Monitoring",
                                       height=45, font=ctk.CTkFont(size=15, weight="bold"),
                                       command=self.toggle)
        self.start_btn.pack(fill="x", padx=20, pady=10)

        stats = ctk.CTkFrame(panel, corner_radius=10)
        stats.pack(fill="x", padx=20, pady=20)
        self.mode_lbl = ctk.CTkLabel(stats, text="Mode: —",
                                     font=ctk.CTkFont(size=14, weight="bold"))
        self.mode_lbl.pack(anchor="w", padx=15, pady=(12, 4))
        self.alerts_lbl = ctk.CTkLabel(stats, text="Alerts: 0")
        self.alerts_lbl.pack(anchor="w", padx=15, pady=4)
        self.timer_lbl = ctk.CTkLabel(stats, text="Session: 0 min")
        self.timer_lbl.pack(anchor="w", padx=15, pady=(4, 12))

        self.status_lbl = ctk.CTkLabel(panel, text="Ready", text_color="gray")
        self.status_lbl.pack(side="bottom", pady=20)

        self.video_lbl = ctk.CTkLabel(self, text="Camera off\nPress Start",
                                      font=ctk.CTkFont(size=18), text_color="gray")
        self.video_lbl.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def reset_state(self):
        self.closed_start = None
        self.last_face_time = time.time()
        self.last_eyes_seen = time.time()
        self.ever_seen_eyes = False
        self.last_alert_time = 0
        self.alerts_count = 0
        self.session_start = time.time()

    def set_alert(self):
        text = self.alert_entry.get().strip() or "انتبه! استيقظ، أنتَ تشعر بالنعاس"
        self.status_lbl.configure(text="Generating voice...", text_color="orange")
        self.update()
        try:
            gTTS(text=text, lang="ar").save("alert_ar.mp3")
            self.alert_sound = pygame.mixer.Sound("alert_ar.mp3")
            self.status_lbl.configure(text="Alert voice updated ✓", text_color="green")
        except Exception:
            self.status_lbl.configure(text="Need internet for voice", text_color="red")

    def toggle(self):
        if self.running:
            self.running = False
            self.start_btn.configure(text="▶  Start Monitoring")
            self.status_lbl.configure(text="Stopped", text_color="gray")
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.reset_state()
            self.running = True
            self.start_btn.configure(text="⏹  Stop Monitoring")
            self.status_lbl.configure(text="Monitoring...", text_color="green")
            threading.Thread(target=self.loop, daemon=True).start()

    def fire_alert(self):
        now = time.time()
        if now - self.last_alert_time > ALERT_COOLDOWN:
            if self.alert_sound:
                self.alert_sound.play()
            self.last_alert_time = now
            self.alerts_count += 1

    def loop(self):
        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            gray = enhance(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            h, w = frame.shape[:2]
            mode = "SEARCHING..."

            if results.multi_face_landmarks:
                self.last_face_time = now
                lm = results.multi_face_landmarks[0].landmark
                if is_face_covered(gray, lm, w, h):
                    mode = "NIQAB MODE"
                    self.run_eyes(frame, gray, now)
                else:
                    mode = "FACE MODE"
                    probs = []
                    for pts in [LEFT_EYE, RIGHT_EYE]:
                        p, box = eye_crop_classify(gray, lm, pts, w, h)
                        if p is not None:
                            probs.append(p)
                            x1, y1, x2, y2 = box
                            c = (0, 255, 0) if p > 0.5 else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                    if probs:
                        if sum(probs)/len(probs) <= 0.5:
                            if self.closed_start is None:
                                self.closed_start = now
                            if now - self.closed_start >= CLOSED_TIME_LIMIT:
                                cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (20, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                                self.fire_alert()
                        else:
                            self.closed_start = None
            elif now - self.last_face_time > FACE_LOST_LIMIT:
                mode = "EYES-ONLY MODE"
                self.run_eyes(frame, gray, now)

            self.mode_lbl.configure(text=f"Mode: {mode}")
            self.alerts_lbl.configure(text=f"Alerts: {self.alerts_count}")
            self.timer_lbl.configure(text=f"Session: {int((now-self.session_start)//60)} min")

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(720, 480))
            self.video_lbl.configure(image=ctk_img, text="")
            self.video_lbl.image = ctk_img

        if self.cap:
            self.cap.release()
        self.video_lbl.configure(image=None, text="Camera off\nPress Start")

    def run_eyes(self, frame, gray, now):
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05,
                                            minNeighbors=4, minSize=(25, 25))
        open_found = False
        for (ex, ey, ew, eh) in eyes:
            p = classify_eye(gray, ex, ey, ew, eh)
            o = p > 0.5
            c = (0, 255, 0) if o else (0, 0, 255)
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), c, 2)
            cv2.putText(frame, f"{'OPEN' if o else 'CLOSED'} {p:.2f}",
                        (ex, ey-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            if o:
                open_found = True
        if open_found:
            self.last_eyes_seen = now
            self.ever_seen_eyes = True
        if self.ever_seen_eyes and now - self.last_eyes_seen >= NIQAB_CLOSED_LIMIT:
            cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            self.fire_alert()

    def on_close(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    DrowsinessApp().mainloop()
