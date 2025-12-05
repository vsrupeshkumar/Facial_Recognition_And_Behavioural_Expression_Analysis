import cv2
import numpy as np
from ultralytics import YOLO
import time

# Try to import FER, handle failure gracefully
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    print("Warning: FER module not found. Emotion detection will be disabled.")
    FER_AVAILABLE = False

class SystemDetector:
    def __init__(self):
        print("Initializing detectors...")
        
        # Emotion Detector
        if FER_AVAILABLE:
            # mtcnn=False uses OpenCV Haar Cascade which is faster
            try:
                self.emotion_detector = FER(mtcnn=False)
            except Exception as e:
                print(f"Error init FER: {e}")
                self.emotion_detector = None
        else:
            self.emotion_detector = None
            
        # Object Detector (YOLOv8 nano for speed)
        # We assume YOLO is installed as it is more robust, but good to be safe
        try:
            self.object_model = YOLO('yolov8n.pt')
            self.suspicious_classes = [67] # cell phone
            self.class_names = self.object_model.names
        except Exception as e:
            print(f"Error init YOLO: {e}")
            self.object_model = None

    def process_frame(self, frame):
        alerts = []
        
        # 1. Object Detection
        if self.object_model:
            results = self.object_model(frame, verbose=False, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in self.suspicious_classes:
                        # Draw Red Box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label = f"SUSPICIOUS: {self.class_names[cls]}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        alerts.append(label)

        # 2. Emotion Detection
        if self.emotion_detector:
            try:
                # FER returns a list of dictionaries
                analysis = self.emotion_detector.detect_emotions(frame)
                for face in analysis:
                    (x, y, w, h) = face['box']
                    emotions = face['emotions']
                    dominant_emotion = max(emotions, key=emotions.get)
                    score = emotions[dominant_emotion]
                    
                    # Draw Face Box (Green)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Text
                    text = f"{dominant_emotion}: {score:.2f}"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except Exception as e:
                pass # Squelch errors during inference
        else:
            # Fallback: Just detect faces with Haar Cascade if available?
            # Or just skip.
            pass

        return frame, alerts
