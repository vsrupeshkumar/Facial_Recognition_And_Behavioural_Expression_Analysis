from flask import Flask, render_template, Response, jsonify
import cv2
from detector import SystemDetector
import threading
import time
import os

app = Flask(__name__)

# Global variables
output_frame = None
lock = threading.Lock()
detector = None
latest_alerts = []
last_alert_time = 0

def initialize_detector():
    global detector
    if detector is None:
        try:
            detector = SystemDetector()
        except Exception as e:
            print(f"Error initializing detector: {e}")

def capture_feed():
    global output_frame, latest_alerts, last_alert_time, detector
    
    # Initialize detector in the thread
    initialize_detector()
    
    # Open Webcam
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read from camera.")
            time.sleep(1)
            continue
        
        if detector:
            try:
                processed_frame, alerts = detector.process_frame(frame)
                
                # Update global frame
                with lock:
                    output_frame = processed_frame.copy()
                    
                # Update alerts (debounce/logic)
                if alerts:
                    current_time = time.time()
                    # Only update alerts every second or so to avoid spam, or keep a history
                    if current_time - last_alert_time > 1.0:
                        latest_alerts = alerts
                        last_alert_time = current_time
                else:
                    if time.time() - last_alert_time > 2.0:
                        latest_alerts = [] # Clear alerts after 2 seconds of silence

            except Exception as e:
                print(f"Error processing frame: {e}")
                with lock:
                    output_frame = frame 
        else:
            with lock:
                output_frame = frame
        
        # Sleep slightly to prevent CPU hogging
        time.sleep(0.01)

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
            
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global latest_alerts
    return jsonify({
        'status': 'active',
        'alerts': latest_alerts
    })

if __name__ == '__main__':
    # Start the camera thread
    t = threading.Thread(target=capture_feed)
    t.daemon = True
    t.start()
    
    # Run Flask App
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
