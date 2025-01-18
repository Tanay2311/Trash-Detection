import os
import cv2
import numpy as np
from tqdm import tqdm  # For the progress bar

# Load YOLOv3 model
yolo_net = cv2.dnn.readNet("/Users/tanay/Downloads/MP_Final/yolov3.weights", "/Users/tanay/Downloads/MP_Final/yolov3.cfg")

# Load class labels
yolo_classes = []
with open('/Users/tanay/Downloads/MP_Final/coco.names', 'r') as f:
    yolo_classes = f.read().splitlines()

# Function to perform YOLO object detection on a single image
def perform_yolo_detection(img, confidence_threshold=0.7, save_dir=None):
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    output_layers_names = yolo_net.getUnconnectedOutLayersNames()
    layer_outputs = yolo_net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

    if len(indexes) > 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 3
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(yolo_classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = tuple(int(c) for c in colors[i])

            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(img, f"{label} {confidence}", (x, y - 5), font, 0.5, (255, 255, 255), 1)

            # Log detection results
            with open("detection_log.txt", "a") as log_file:
                log_file.write(f"{label},{confidence},{x},{y},{w},{h}\n")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, img)

        return img
    else:
        return None

# Load images from the specified directory
image_dir = '/Users/tanay/Downloads/MP_Final/dataset/images'
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

# Directory to save images with detections
save_dir = '/Users/tanay/Downloads/MP_Final/detected_images'

# Iterate over your dataset images with a progress bar
for image_path in tqdm(image_paths, desc="Processing images"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image file '{image_path}'")
        continue

    yolo_detected_image = perform_yolo_detection(img, save_dir=save_dir)

cv2.destroyAllWindows()
