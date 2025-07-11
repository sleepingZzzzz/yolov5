# Course Design: Traffic Flow Analysis and Signal Optimization 

# Using YOLOv 5

# Objective 

1. Master the YOLOv 5 object detection method and its application in traffic 

monitoring. 

2. Learn multi-object tracking (MOT) techniques for vehicle counting. 

3. Analyze the impact of traffic signal control strategies on reducing idling energy 

consumption. 

# Project Links 

 YOLOv 5 Official Code: https://github.com/ultralytics/yolov5 

 Energy Consumption 

Reference: https://www.sciencedirect.com/science/article/pii/S1361920920308597 

# Experimental Environment 

1. Install Anaconda3 and create a virtual environment: 

conda create -n yolov 5_traffic python=3.8 

conda activate yolov5_traffic 

git clone https://github.com/ultralytics/yolov5.git 

cd yolov5 

pip install -r requirements.txt 

2. Modify requirements.txt: 

 Ensure OpenCV version is 4.1.2.30 for compatibility: opencv-python==4.1.2.30 

# Experimental Steps 

1. Dataset Preparation 

Dataset : UA-DETRAC (Vehicle Detection and Tracking Dataset) 

1.1. Directory Structure 

├── yolov 5/

└── datasets/ 

└── UA-DETRAC/ ├── images/ # Extracted video frames (JPEG) 

│ ├── train/ 

│ └── val/ 

└── labels/ # YOLO-format labels 

├── train/ 

└── val/ 

1.2. YAML Configuration (data/ua-detrac.yaml) 

path: ./datasets/UA-DETRAC 

train: images/train 

val: images/val 

nc: 1 # Only detect vehicles 

names: ['vehicle'] 

1.3. Label Conversion 

Use a script to convert UA-DETRAC ’s XML labels to YOLO format: 

# xml_to_yolo.py (example snippet) 

def convert(size, box): 

x_center = (box[0] + box[2]) / 2.0 / size[0] 

y_center = (box[1] + box[3]) / 2.0 / size[1] 

width = (box[2] - box[0]) / size[0] 

height = (box[3] - box[1]) / size[1] 

return [x_center, y_center, width, height] 

2. Model Training & Testing 

2.1. Download Pre-trained Weights 

python detect.py --weights yolov5s.pt --source data/ua-detrac.yaml --img 640 --conf 

0.25 

2.2. Training Command 

python train.py --data data/ua-detrac.yaml --weights yolov5s.pt --batch-size 16 --

img 640 --epochs 50 --device 0

2.3. Testing Command 

python test.py --weights runs/train/exp/weights/best.pt --data ua-detrac.yaml --img 

640 --task val --device 03. Traffic Flow Analysis & Energy Optimization 

3.1. Multi-Object Tracking (DeepSORT Integration) 

Modify detect.py to integrate DeepSORT for trackin g:

from deep_sort_realtime.deepsort_tracker import DeepSort 

tracker = DeepSort(max_age=30) 

tracks = tracker.update(detections) 

3.2. Fuel Consumption Estimation 

Calculate idling fuel waste using the IDM model (simplified): 

def calc_fuel_consumption(idle_time_sec): 

# Assumption: Idling consumes 0.2 liters/minute 

return idle_time_sec / 60 * 0.2 # Returns : float (liters )

3.3. Traffic Signal Simulation (PyGame) 

Dynamically adjust green light duration based on vehicle count: 

green_time = base_time + 5 * len(current_lane_vehicles) 

# Submission Requirements 

1. Directory Structure :

├── StudentID_Name/ 

│ ├── yolov 5/ # Full code (including trained best.pt) 

│ ├── report.docx # Lab report (with mAP@0.5 & energy savings) 

│ ├── presentation.pptx # Presentation slides 

│ └── demo.mp4 # Traffic flow visualization demo 

2. Grading Criteria:  

> 

Basic : Vehicle detection (mAP@0.5 ≥ 0.7 on validation set ).  

> 

Advanced : Tracking + energy analysis (must show % reduction in idling time 

compared to static green light duration ).