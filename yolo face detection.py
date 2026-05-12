

import cv2


model = cv2.dnn.readNetFromCaffe(
    r"C:\Users\sneha\spyder\deploy.prototxt",
    r"C:\Users\sneha\spyder\res10_300x300_ssd_iter_140000.caffemodel"
)


webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Advanced Face Detection Started!")
print("Press 'd' to exit.")

while True:


    ret, frame = webcam.read()

    if not ret:
        print("Failed to grab frame.")
        break


    (h, w) = frame.shape[:2]

    
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    
    model.setInput(blob)
    detections = model.forward()

    person_no = 1
    people_count = 0

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:

            people_count += 1

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

          
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

         
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

  
    cv2.putText(
        frame,
        f'Total People: {people_count}',
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

 
    cv2.imshow("Advanced Real-Time Face Detection", frame)

 
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break


webcam.release()
cv2.destroyAllWindows()

print("Webcam closed.")
cv2.destroyAllWindows()
