

from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")


webcam = cv2.VideoCapture(0)


webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


if not webcam.isOpened():
    print("Error opening webcam")
    exit()

print("YOLO AI Detection Started")
print("Press 'd' to exit")


cv2.namedWindow(
    "YOLO Real-Time Detection",
    cv2.WND_PROP_FULLSCREEN
)

cv2.setWindowProperty(
    "YOLO Real-Time Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:

   
    ret, frame = webcam.read()

    if not ret:
        break

   
    results = model(frame)

    person_count = 0


    for result in results:

        boxes = result.boxes

        for box in boxes:

            
            confidence = float(box.conf[0])

            
            class_id = int(box.cls[0])

           
            if class_id == 0 and confidence > 0.5:

                person_count += 1

                #
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

              
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

           
                cv2.putText(
                    frame,
                    f'Person {person_count}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

  
    cv2.putText(
        frame,
        f'Total People in the frame: {person_count}',
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

   
    cv2.imshow(
        "YOLO Real-Time Detection",
        frame
    )

  
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break


webcam.release()
cv2.destroyAllWindows()