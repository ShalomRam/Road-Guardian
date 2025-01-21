from flask import Flask, render_template, request, jsonify, url_for, redirect
import os
import subprocess
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload and result folders
UPLOAD_FOLDER = 'E:/Roadify/data/uploads'
RESULT_FOLDER = 'E:/Roadify/data/final_result'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('sh.html')

@app.route('/web.html')
def live_feed():
    return render_template('web.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Determine if file is an image or video
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        # Run image.py with the image path
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'output_image.jpg')
        subprocess.run(['python', 'image.py', file_path, result_path])

        return jsonify({
            "status": "success",
            "output_image": url_for('static', filename=f'{RESULT_FOLDER}/output_image.jpg')
        })

    elif file.filename.lower().endswith('.mp4'):
        # Run camera_video.py with the video path
        result_video_path = os.path.join(app.config['RESULT_FOLDER'], 'output_video.mp4')
        subprocess.run(['python', 'camera_video.py', file_path, result_video_path])

        return jsonify({
            "status": "success",
            "output_video": url_for('static', filename=f'{RESULT_FOLDER}/output_video.mp4')
        })

    else:
        return jsonify({"status": "error", "message": "Invalid file type. Only JPG, PNG, and MP4 are supported."}), 400

if __name__ == "__main__":
    app.run(debug=True)
