import os
import cv2
import numpy as np

# Load YOLOv3 model
yolo_net = cv2.dnn.readNet("/Users/tanay/Downloads/MP_Final/yolov3.weights", "/Users/tanay/Downloads/MP_Final/yolov3.cfg")

# Load class labels
yolo_classes = []
with open('/Users/tanay/Downloads/MP_Final/coco.names', 'r') as f:
    yolo_classes = f.read().splitlines()

# Function to perform YOLO object detection on a single image
def perform_yolo_detection(img):
    height, width, _ = img.shape  # Get image dimensions

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    # Get YOLO output
    output_layers_names = yolo_net.getUnconnectedOutLayersNames()
    layer_outputs = yolo_net.forward(output_layers_names)

    # Parse YOLO output
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

    # Draw bounding boxes and labels on the image if objects are detected
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
            cv2.putText(img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)

        return img
    else:
        return None

# Load images from the specified directory
#image_dir = '/Users/tanay/Downloads/TACO/data/Images_1/images'
image_dir='/Users/tanay/Downloads/MP_Final/dataset/images'
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

# Iterate over your dataset images
for image_path in image_paths:
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image file '{image_path}'")
        continue

    # Perform YOLO object detection on the image
    yolo_detected_image = perform_yolo_detection(img)

    # Display the image with bounding boxes if objects are detected
    if yolo_detected_image is not None:
        cv2.imshow('YOLO Object Detection', yolo_detected_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
