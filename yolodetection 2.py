from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Advanced YOLO Detection Started!")
print("Press 'd' to exit.")

while True:

    # Read frame
    ret, frame = webcam.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Get frame size
    (h, w) = frame.shape[:2]

    # YOLO detection
    results = model(frame)

    person_no = 1
    people_count = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            # Detect PERSON only
            if class_id == 0 and confidence > 0.6:

                people_count += 1

                # Get box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
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

    # Show total people count
    cv2.putText(
        frame,
        f'Total People: {people_count}',
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Show output
    cv2.imshow(
        "Advanced Real-Time YOLO Detection",
        frame
    )

    # Exit when d is pressed
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

print("Webcam closed.")