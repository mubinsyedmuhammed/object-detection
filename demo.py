import cv2

thres = 0.3  # Lowered threshold for better detection

# Capture video from the first available camera (index 0)
cap = cv2.VideoCapture(0)

# Set resolution and brightness (optional)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
cap.set(10, 70)   # Brightness

# Load class names from COCO dataset
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Model configuration and weights files
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the detection model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(640, 640)  # Increased input size for better accuracy
net.setInputScale(1.0 / 255)  # Normalizing the pixel values
net.setInputMean((0, 0, 0))   # Reset the mean values
net.setInputSwapRB(True)      # Swap R and B channels

# List of desired class IDs from COCO dataset (e.g., person, car, bicycle, etc.)
desired_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Class IDs for person, bicycle, car, etc.

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    # Perform object detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs.flatten(), thres, 0.4)  # NMS threshold set to 0.4
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            confidence = confs[i]
            classId = classIds[i]

            # Filter only the desired class IDs
            if classId in desired_class_ids:
                # Draw rectangle and label the detected object
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Output", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
