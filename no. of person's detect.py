import cv2

# Load DNN face detection model
model = cv2.dnn.readNetFromCaffe(
    r"C:\Users\sneha\spyder\deploy.prototxt",
    r"C:\Users\sneha\spyder\res10_300x300_ssd_iter_140000.caffemodel"
)

# Start webcam
webcam = cv2.VideoCapture(0)

# Improve camera quality
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
webcam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Check webcam
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Advanced Face Detection Started!")
print("Press 'd' to exit.")

# FULLSCREEN WINDOW
cv2.namedWindow(
    "Advanced Real-Time Face Detection",
    cv2.WND_PROP_FULLSCREEN
)

cv2.setWindowProperty(
    "Advanced Real-Time Face Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:

    # Read frame
    ret, frame = webcam.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Smooth image for cleaner output
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # Slight brightness improvement
    frame = cv2.convertScaleAbs(
        frame,
        alpha=1.1,
        beta=8
    )

    # Get frame size
    (h, w) = frame.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    # Pass blob through network
    model.setInput(blob)
    detections = model.forward()

    person_no = 1

    # Loop through detections
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # LOWER confidence to detect small faces in photos/screens
        if confidence > 0.40:

            # Get face box coordinates
            box = detections[0, 0, i, 3:7] * [w, h, w, h]

            (x1, y1, x2, y2) = box.astype("int")

            # Prevent negative values
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            # Ignore very tiny false detections
            if (x2 - x1) > 40 and (y2 - y1) > 40:

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Show person number
                cv2.putText(
                    frame,
                    f'Person {person_no}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                person_no += 1

    # Show FULLSCREEN output
    cv2.imshow(
        "Advanced Real-Time Face Detection",
        frame
    )

    # Exit when d is pressed
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

print("Webcam closed.")