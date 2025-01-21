# Importing necessary libraries
import cv2
import os
import sys

# Load and process image (rest of your code remains the same)
file_path = sys.argv[1]
result_path = sys.argv[2]

# Initialize counters for accuracy calculation
correct_predictions = 0
total_predictions = 0

# Define ground truth for each image (0 for plain, 1 for pothole)
# Here you need to specify whether each image in your set is actually a pothole or plain road.
# This is just a placeholder. Replace this with actual ground truth data for each test image.
ground_truth_label = 1  # Set to 1 if the image contains a pothole, 0 if it's a plain road

# Reading test image
i=0
for i in range(0,100):
    img = cv2.imread(file_path) # Update with the correct image path
    if img is None:
        raise ValueError("Image could not be loaded. Check if the image file is valid.")

    # Reading label names from obj.names file
    with open(os.path.join("project_files", 'obj.names'), 'r') as f:
        classes = f.read().splitlines()

    # Importing model weights and config file
    net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    # Perform detection on the image
    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

    # Check if the detection matches the ground truth
    predicted_label = 1 if len(classIds) > 0 else 0  # Assume 1 (pothole) if any object detected, 0 otherwise
    total_predictions += 1
    if predicted_label == ground_truth_label:
        correct_predictions += 1

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100

    # Display accuracy on the image
    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                    color=(199, 36, 177), thickness=2) #Box ka color RGB form me
        cv2.putText(img, f"{classes[classId]}: {score:.2f}", (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (68, 214, 44), 2) #Text ka color RGB Form me

    # Display the real-time accuracy on the image
    #v2.putText(img, f"Accuracy: {accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (33, 46, 82), 2)

    # Show the result
    cv2.imshow("Pothole Detection", img)
    result = cv2.imwrite("E:/Roadify/data/final_result/output_image.jpg", img)  # Save the image with accuracy displayed
    cv2.waitKey(0)

    cv2.imwrite(result_path, img)

    # Print final accuracy
    #print(f"Final Accuracy: {accuracy:.2f}%")