# Real-Time AI-Based Multi-Person Detection and Counting System Using YOLOv8 and OpenCV

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![YOLOv8](https://img.shields.io/badge/YOLOv8-111111?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Deep Learning](https://img.shields.io/badge/DeepLearning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-5C2D91?style=for-the-badge)
![Artificial Intelligence](https://img.shields.io/badge/Artificial%20Intelligence-blue?style=for-the-badge)
![Object Detection](https://img.shields.io/badge/Object%20Detection-green?style=for-the-badge)
![Webcam](https://img.shields.io/badge/Webcam-red?style=for-the-badge)

---
# Project Overview



The Real-Time AI-Based Multi-Person Detection and Counting System Using YOLOv8 and OpenCV is an advanced computer vision project that detects and counts multiple people instantly through a live webcam feed. Developed using Python, YOLOv8, OpenCV, and Deep Learning technology, the system performs real-time human detection with high speed, improved accuracy, and intelligent object recognition capabilities.

The project continuously captures live video frames from the webcam and uses the YOLOv8 Deep Learning model to identify human objects in real time. Detected persons are highlighted using dynamic bounding boxes and labels such as Person 1, Person 2, and more, along with the total people count displayed on the screen.

Unlike traditional detection techniques, YOLOv8 uses advanced Deep Learning and object detection algorithms that provide faster detection, better long-distance recognition, improved small-object detection, and higher real-time performance. The system can accurately detect people even in crowded environments, different lighting conditions, and complex backgrounds.

This project demonstrates the practical implementation of Artificial Intelligence, Deep Learning, and Computer Vision in developing intelligent real-world applications. It can be used in smart surveillance systems, crowd monitoring, security systems, smart attendance systems, traffic monitoring, and AI-powered automation applications.

---

# Why YOLOv8 Was Used Instead of Traditional OpenCV DNN

Initially, the project was developed using traditional OpenCV Deep Neural Network (DNN) detection techniques. While the system was able to perform basic real-time detection, several limitations were observed during testing in real-world conditions.

## Problems Identified in the Initial OpenCV DNN Model

- Inaccurate long-distance detection
- Difficulty detecting small faces/persons
- Missed detections in group scenes
- Reduced accuracy in low lighting conditions
- Slower performance in crowded environments
- Less reliable real-time detection

To overcome these challenges, the system was upgraded using the YOLOv8 Deep Learning model.

## Improvements Achieved Using YOLOv8

- Faster real-time detection
- Improved long-distance person detection
- Better small-object and small-person detection
- Higher detection accuracy
- Improved crowd and multi-person detection
- Reduced false detections
- Better real-time AI performance

The YOLOv8 model significantly enhanced the overall efficiency, speed, and reliability of the system, making it more suitable for intelligent real-world AI surveillance and monitoring applications.

---

# Why YOLOv8 is Better Than Traditional OpenCV DNN

| Feature | OpenCV DNN | YOLOv8 |
|---|---|---|
| Detection Speed | Medium | Very Fast |
| Real-Time Performance | Good | Excellent |
| Long-Distance Detection | Weak | Strong |
| Small Object Detection | Limited | Highly Accurate |
| Crowd Detection | Less Accurate | More Accurate |
| Detection Accuracy | Moderate | High |
| Multiple Object Detection | Limited | Advanced |
| False Detections | More | Less |
| AI Capability | Basic DNN | Advanced Deep Learning |
| Scalability | Limited | High |

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam handling and image processing |
| YOLOv8 | Real-time object/person detection |
| Deep Learning | Intelligent AI-based detection |
| Computer Vision | Image and video analysis |
| Ultralytics YOLO | YOLOv8 framework and model support |
| Webcam | Live video input source |

---

# Techniques Used

| Technique | Purpose |
|---|---|
| Object Detection | Detect humans in video frames |
| Deep Neural Networks (DNN) | AI-based intelligent detection |
| Real-Time Video Processing | Continuous webcam frame analysis |
| Bounding Box Detection | Highlight detected persons |
| Confidence Scoring | Filter weak detections |
| Image Processing | Improve frame analysis |
| Multi-Person Detection | Detect multiple humans simultaneously |
| AI-Based Recognition | Intelligent object identification |

---

# Features

- Real-time human detection
- AI-based people counting
- Fullscreen live webcam feed
- Multiple person detection
- Bounding boxes with labels
- Long-distance detection
- Small object detection
- Real-time AI processing
- High-speed detection performance
- Dynamic people count display

---

# Applications

- Smart Surveillance Systems
- Crowd Monitoring Systems
- Smart Security Systems
- AI-Based Attendance Systems
- Traffic Monitoring
- Human Detection Systems
- Smart City Applications
- Real-Time Monitoring Systems

---

# Future Enhancements

- Face recognition with names
- Attendance management system
- GPU acceleration
- Mobile camera integration
- Person tracking IDs
- Motion tracking
- Video recording system
- AI analytics dashboard

---

# Installation

## Install Required Libraries

```bash
pip install ultralytics
pip install opencv-python
