# Importing necessary libraries
import cv2 as cv
import time
import geocoder
import os
import sys

# Load and process video (rest of your code remains the same)
video_path = sys.argv[1]
output_video_path = sys.argv[2]

# Check if necessary files exist before loading
def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found. Check the file path.")

# Check if the result directory exists, create it if it doesn't
result_path = "pothole_coordinates"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Reading label names from obj.names file
class_name = []
obj_names_path = os.path.join("project_files", 'obj.names')
check_file_exists(obj_names_path)
with open(obj_names_path, 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Importing model weights and config file, and defining the model parameters
weights_path = 'project_files/yolov4_tiny.weights'
config_path = 'project_files/yolov4_tiny.cfg'
check_file_exists(weights_path)
check_file_exists(config_path)

net1 = cv.dnn.readNet(weights_path, config_path)
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Defining the video source (0 for camera or file name for video)
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Could not open video file.")

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# Lowering the frame rate for better processing
processing_fps = 10  # Adjust this value for optimal processing
frame_skip = int(fps // processing_fps)

# Video writer setup
result = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), processing_fps, (width, height))

# Parameters for detection and saving results
g = geocoder.ip('me')
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
processed_frames = 0
i = 0

# Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to achieve the desired processing frame rate
    if frame_counter % frame_skip != 0:
        frame_counter += 1
        continue

    frame_counter += 1
    processed_frames += 1

    # Detection on frame
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = width * height

        # Drawing detection boxes on frame
        if len(scores) != 0 and scores[0] >= 0.7:
            if (recarea / area) <= 0.1 and box[1] < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, f"{round(scores[0] * 100, 2)}% {label}", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display FPS on frame
    elapsed_time = time.time() - starting_time
    fps = processed_frames / elapsed_time
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save and show the frame
    result.write(frame)
    cv.imshow('Pothole Detection', frame)

    # Exit on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
result.release()
cv.destroyAllWindows()

print(f"Processed video saved as: {output_video_path}")
