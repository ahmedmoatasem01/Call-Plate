from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from detect_and_recognize import detect_number_plates, recognize_number_plates
from ultralytics import YOLO
from easyocr import Reader
import cv2
import numpy as np
import base64
import os
import logging
import tempfile
import shutil

import base64




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and reader once
model = YOLO("runs/detect/train/weights/best.pt")
reader = Reader(['en'], gpu=True)

def process_video(file_path, model, reader, output_folder):
    video_cap = cv2.VideoCapture(file_path)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f"processed_{os.path.basename(file_path)}"
    output_path = os.path.join(output_folder, output_filename)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    results = []
    cropped_paths = []

    while True:
        success, frame = video_cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_copy = image_rgb.copy()

        number_plate_list = detect_number_plates(image_rgb, model)
        if number_plate_list:
            number_plate_list = recognize_number_plates(image_rgb, reader, number_plate_list)
            for item in number_plate_list:
                box = item[0]
                text = item[1] if len(item) > 1 else ""
                xmin, ymin, xmax, ymax = box
                cropped = image_copy[ymin:ymax, xmin:xmax]
                crop_filename = f"crop_{xmin}_{ymin}_{xmax}_{ymax}_{output_filename}"
                crop_path = os.path.join(output_folder, crop_filename)
                cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                cropped_paths.append(f"/static/uploads/{crop_filename}")
                cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image_rgb, text, (xmin, ymax + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                results.append({'text': text, 'crop': cropped_paths[-1]})

            writer.write(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        else:
            writer.write(frame)

    video_cap.release()
    writer.release()

    return output_path, results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    file_ext = filename.rsplit('.', 1)[1].lower()
    if file_ext in ['jpg', 'jpeg', 'png']:
        # IMAGE PROCESSING
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_copy = image_rgb.copy()

        number_plate_list = detect_number_plates(image_rgb, model)
        results = []
        cropped_paths = []

        if number_plate_list:
            number_plate_list = recognize_number_plates(filepath, reader, number_plate_list)
            for item in number_plate_list:
                box = item[0]
                text = item[1] if len(item) > 1 else ""
                xmin, ymin, xmax, ymax = box
                cropped = image_copy[ymin:ymax, xmin:xmax]
                crop_filename = f"crop_{xmin}_{ymin}_{xmax}_{ymax}_{filename}"
                crop_path = os.path.join(UPLOAD_FOLDER, crop_filename)
                cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                cropped_paths.append(f"/static/uploads/{crop_filename}")
                cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image_rgb, text, (xmin, ymax + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                results.append({'text': text, 'crop': cropped_paths[-1]})

            # Save processed image
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(UPLOAD_FOLDER, processed_filename)
            cv2.imwrite(processed_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            with open(processed_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode('utf-8')


            return jsonify({
                'success': True,
                'type': 'image',
                'original_image': f"/static/uploads/{filename}",
                'processed_image': f"/static/uploads/{processed_filename}",
                'results': results,
                'image': b64_string
            })
        else:
            return jsonify({
                'success': False,
                'type': 'image',
                'message': 'No number plate detected.',
                'original_image': f"/static/uploads/{filename}"
            })

    elif file_ext in ['mp4', 'mov', 'avi', 'mkv']:
        # VIDEO PROCESSING
        processed_video_path, results = process_video(filepath, model, reader, UPLOAD_FOLDER)
        return jsonify({
            'success': True,
            'type': 'video',
            'original_video': f"/static/uploads/{filename}",
            'processed_video': f"/static/uploads/{os.path.basename(processed_video_path)}",
            'results': results,
            'message': 'Video processed. Download or view the result below.'
        })

    else:
        return jsonify({'error': 'Unsupported file type.'}), 400

@app.route('/static/uploads/<filename>')
def serve_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, mimetype='video/mp4')

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 