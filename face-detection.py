import cv2

# Load the DNN face detection model
model_path = "opencv_face_detector_uint8.pb"
config_path = "opencv_face_detector.pbtxt"

# Initialize the DNN model
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# Set up the video capture for webcam feed
cap = cv2.VideoCapture(0)

face_counter=0

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Prepare the frame as input to the DNN
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)

    # Run the model to detect faces
    detections = net.forward()

    # Process detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter detections by confidence level
        if confidence > 0.5:  # Adjust threshold as needed
            # Extract the bounding box coordinates
            box = detections[0, 0, i, 3:7] * [width, height, width, height]
            (x, y, x1, y1) = box.astype("int")
            
            # Draw the bounding box and confidence score on the frame
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            text = f"{confidence * 100:.2f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            # Crop and save the detected face region
            face = frame[y:y1, x:x1]
            # face_filename = f"cropped_img/detected_face_{face_counter}.jpg"
            # cv2.imwrite(face_filename, face)
            # print(f"Saved {face_filename}")
            # face_counter += 1

    # Display the frame with detections
    cv2.imshow("Face Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
