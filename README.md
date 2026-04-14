# Smart Traffic Surveillance using YOLOv8  
## AI-Powered Vehicle Detection, Tracking & Speed Violation System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Computer Vision](https://img.shields.io/badge/Field-Computer%20Vision-orange)

---

## 🎥 Project Demonstration

Watch the full system analysis:

https://youtu.be/d_OVW6mwGVc

The demonstration shows real-time vehicle detection, persistent tracking IDs, and automated speed violation detection.

---

## 🚀 Project Overview

This project implements a computer vision traffic monitoring system capable of detecting and tracking vehicles in real-time road footage.

Using YOLOv8 Nano and OpenCV, the system performs frame-level vehicle detection and tracking while estimating speed based on calibrated distance measurements.

Vehicles exceeding the speed threshold trigger a simulated e-challan event, which is logged automatically.

---

## 🏗 System Architecture

The traffic monitoring engine is built using a modular pipeline:

### 1. Video Processing Layer
Traffic footage is processed frame-by-frame using OpenCV.

### 2. Detection Layer
YOLOv8 Nano detects vehicles belonging to COCO classes:
- Car
- Motorcycle
- Bus
- Truck

### 3. Tracking Layer
The tracking system assigns persistent IDs to vehicles using YOLO tracking with `persist=True`.

This ensures each vehicle maintains the same ID across frames.

### 4. Speed Estimation Layer
Vehicle speed is calculated by measuring the frame difference between two predefined lines representing a calibrated real-world distance.

Speed is computed using:

Distance / Time → km/h conversion.

### 5. Violation Detection
Vehicles exceeding the speed threshold trigger a violation event.

### 6. Reporting Layer
Violations are logged into a structured dataset using Pandas.

---

## ✨ Key Features

• Real-time vehicle detection using YOLOv8 Nano  
• Multi-object tracking with persistent IDs  
• Speed estimation using calibrated frame geometry  
• Automated violation detection logic  
• CSV-based analytical reporting  
• Memory-safe tracking using stale ID eviction  

---

## 📂 Project Structure

```text
traffic-surveillance-ai/
├── data/
│   ├── input/               # Source video files for processing
│   └── output/              # Generated e-evidence and reports
├── models/
│   └── yolov8n.pt           # Pre-trained YOLOv8 weights
├── src/
│   ├── main.py              # Core execution engine
│   └── utils.py             # Geometry and speed calculation helpers
├── requirements.txt         # Production dependencies
├── .gitignore               # Excludes large binaries and caches
└── README.md                # Project documentation
```
### 🎥 Produced Evidence
- **File:** `data/output/echallan_evidence.mp4`
- **Description:** A high-definition reconstructed video featuring the YOLOv8 annotation layer, persistent tracking IDs, and real-time speed violation "CHALLAN" HUD overlays.

> 📺 **[Watch the Full Demo & Raw Output on YouTube](https://youtu.be/d_OVW6mwGVc)**

### Getting Started 
📁 Project Output
🎥 Produced Evidence
Input Video (Raw):
data/input/traffic_sample.mp4
➜ Original unprocessed traffic footage used as input
Processed Video (E-Challan Output):
data/output/echallan_evidence.mp4
➜ YOLOv8 annotated video with tracking IDs + speed violation overlays
Violation Report (CSV):
data/output/violations_report.csv
➜ Structured log of detected speed violations (vehicle ID, speed, timestamp, penalty status)

 #### Install Dependencies 
   ```bash
   pip install -r requirements.txt 

 Run the Engine 
   python src/main.py
