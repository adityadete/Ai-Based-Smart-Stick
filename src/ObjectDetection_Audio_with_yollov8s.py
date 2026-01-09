import cv2
import time
import threading
import queue
from ultralytics import YOLO

# ================= CONFIGURATION =================
KNOWN_WIDTHS = {"person": 0.5, "window": 1.0, "door": 0.9, "chair": 0.5}
FOCAL_LENGTH = 650 
STABILITY_THRESHOLD = 10 # Must see object for 10 frames before speaking
SILENCE_COOLDOWN = 15.0  # Absolute silence for 15s after speaking

# ================= VOICE SYSTEM =================
voice_queue = queue.Queue()

def voice_thread():
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    while True:
        text = voice_queue.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()
        voice_queue.task_done()

threading.Thread(target=voice_thread, daemon=True).start()

# ================= MODEL =================
model = YOLO("yolov8s-oiv7.pt") 

# ================= STATE MEMORY =================
last_spoken_name = None
last_spoken_status = None
last_spoken_time = 0
last_seen_time = {} # Tracks when an object was last seen
detection_counter = {}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, conf=0.6, verbose=False) # Increased confidence to 0.6
    current_frame_objects = {}

    for box in results[0].boxes:
        raw_name = model.names[int(box.cls[0])].lower()
        name = "person" if "person" in raw_name or "face" in raw_name else raw_name
        
        if name in KNOWN_WIDTHS:
            x1 = box.xyxy[0][0].item()
            x2 = box.xyxy[0][2].item()
            pixel_width = max(1, x2 - x1)
            dist = (KNOWN_WIDTHS[name] * FOCAL_LENGTH) / pixel_width
            current_frame_objects[name] = round(dist, 1)

    # 1. Update Detection Timers
    now = time.time()
    active_object = None
    
    for name, dist in current_frame_objects.items():
        detection_counter[name] = detection_counter.get(name, 0) + 1
        last_seen_time[name] = now # Mark that we see them NOW
        
        if detection_counter[name] >= STABILITY_THRESHOLD:
            active_object = (name, dist)
            break

    # 2. Cleanup: If an object hasn't been seen for 2 seconds, reset its counter
    for name in list(detection_counter.keys()):
        if now - last_seen_time.get(name, 0) > 2.0:
            detection_counter[name] = 0

    # 3. PRECISION SPEAKING LOGIC
    if active_object:
        obj_name, obj_dist = active_object
        
        # Categorize distance with a 0.5m buffer (Hysteresis)
        if obj_dist < 1.5: status = "near"
        elif obj_dist < 4.0: status = "at medium distance"
        else: status = "far"

        # SPEAK ONLY IF:
        # A. Different object name OR
        # B. Same object but distance category changed AND 15s have passed
        is_new_type = (obj_name != last_spoken_name)
        is_new_status = (status != last_spoken_status)
        is_cooldown_over = (now - last_spoken_time > SILENCE_COOLDOWN)

        if (is_new_type and is_cooldown_over) or (is_new_status and is_cooldown_over):
            if voice_queue.empty():
                speech_msg = f"{obj_name} {status}"
                voice_queue.put(speech_msg)
                
                last_spoken_name = obj_name
                last_spoken_status = status
                last_spoken_time = now

    # Debug
    cv2.imshow("Precision Smart Stick", results[0].plot())
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
