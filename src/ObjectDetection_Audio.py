from ultralytics import YOLO
import cv2
import time
import subprocess

# ---------------- SILENT WINDOWS SPEECH ----------------
def speak(text):
    subprocess.Popen(
        [
            "powershell",
            "-Command",
            f"Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
        ],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = YOLO("yolov8n.pt")
print("Model loaded")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

# ---------------- PRESENCE MEMORY ----------------
visible_objects = set()        # objects currently in frame
missing_counter = {}           # {object_name: frames_missing}
MISSING_LIMIT = 15              # frames before considering disappeared

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = model(frame, imgsz=416, conf=0.35)
    annotated = results[0].plot()

    current_objects = set()

    # ---------- DETECTION LOOP ----------
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        area = (x2 - x1) * (y2 - y1)

        # Position
        if cx < w / 3:
            position = "left"
        elif cx > 2 * w / 3:
            position = "right"
        else:
            position = "front"

        sentence = f"{name} {position}"
        current_objects.add(name)

        # ---------- SPEAK ONLY WHEN OBJECT APPEARS ----------
        if name not in visible_objects:
            print("SPEAK:", sentence)
            speak(sentence)
            visible_objects.add(name)
            missing_counter[name] = 0

        else:
            missing_counter[name] = 0  # reset missing count

    # ---------- HANDLE DISAPPEARED OBJECTS ----------
    for name in list(visible_objects):
        if name not in current_objects:
            missing_counter[name] += 1
            if missing_counter[name] > MISSING_LIMIT:
                visible_objects.remove(name)
                del missing_counter[name]   # forgotten → will speak again if reappears

    cv2.imshow("Object Detection with Voice Assistance", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
