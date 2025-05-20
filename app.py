from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
from loader import load_files
import os
import uuid
import base64

app = Flask(__name__)

# Initialize variables
cascade = None
jewelleries = None
current_jewel_index = 0
jewel_key_list = None
jewellery = None
impose = None
mx = my = None
dw = dh = None

# Constants
WIDTH = 720
HEIGHT = 640

def init_system():
    """Initialize the jewellery system by loading models and setting initial jewellery"""
    global jewellery, impose, mx, my, dw, dh, current_jewel_index, jewel_key_list, jewelleries, cascade
    # Load the required files into memory
    cascade, jewelleries = load_files()
    jewel_key_list = list(jewelleries)

    # Set the first jewellery as current jewellery in use
    current_jewel_index = 0
    jewellery = jewelleries[jewel_key_list[current_jewel_index]]
    impose = jewellery["path"]
    mx, my = jewellery["x"], jewellery["y"]
    dw, dh = jewellery["dw"], jewellery["dh"]

def process_frame(frame):
    """Process a video frame by detecting faces and applying jewellery"""
    global jewellery, impose, mx, my, dw, dh
    
    if frame is None:
        return None

    # Resize the frame
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    
    # Detect available faces in frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=3)

    if len(faces) == 1:
        # Extract face location
        x, y, w, h = faces[0]

        # Adjust the jewelery to the face's placement
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        fw, fh = int(w * dw), int(w * dh)
        new_impose = cv2.resize(impose, (fw, fh))

        # Super impose the jewellery over captured frame
        iw, ih, c = new_impose.shape
        for i in range(0, iw):
            for j in range(0, ih):
                if new_impose[i, j][3] != 0:
                    if y + i + h + my < HEIGHT and x + j + mx < WIDTH:
                        frame[y + i + h + my, x + j + mx] = new_impose[i, j]

    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

@app.route('/process_frame', methods=['POST'])
def process_frame_api():
    """Accepts a base64 image, processes it, and returns the processed image as base64."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'No image data'}), 400
    image_data = data['image']
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    try:
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed)
        processed_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'status': 'success', 'processed_image': 'data:image/jpeg;base64,' + processed_b64})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/upload_capture', methods=['POST'])
def upload_capture():
    """Accepts a base64 image from the browser, saves it, and returns status."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'No image data'}), 400
    image_data = data['image']
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    try:
        img_bytes = base64.b64decode(image_data)
        if not os.path.exists('captures'):
            os.makedirs('captures')
        filename = f'captures/capture_{uuid.uuid4()}.jpg'
        with open(filename, 'wb') as f:
            f.write(img_bytes)
        return jsonify({'status': 'success', 'filename': filename})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/next_jewellery')
def next_jewellery():
    """Switch to next jewellery item"""
    global current_jewel_index, jewellery, impose, mx, my, dw, dh
    if current_jewel_index + 1 != len(jewel_key_list):
        current_jewel_index += 1
    else:
        current_jewel_index = 0
    jewellery = jewelleries[jewel_key_list[current_jewel_index]]
    impose = jewellery["path"]
    mx, my = jewellery["x"], jewellery["y"]
    dw, dh = jewellery["dw"], jewellery["dh"]
    return jsonify({"status": "success"})

@app.route('/prev_jewellery')
def prev_jewellery():
    """Switch to previous jewellery item"""
    global current_jewel_index, jewellery, impose, mx, my, dw, dh
    if current_jewel_index - 1 < 0:
        current_jewel_index = len(jewel_key_list) - 1
    else:
        current_jewel_index -= 1
    jewellery = jewelleries[jewel_key_list[current_jewel_index]]
    impose = jewellery["path"]
    mx, my = jewellery["x"], jewellery["y"]
    dw, dh = jewellery["dw"], jewellery["dh"]
    return jsonify({"status": "success"})

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

if __name__ == '__main__':
    init_system()
    app.run(debug=True, host='0.0.0.0')
