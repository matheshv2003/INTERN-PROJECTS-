
 Real-Time Faint Detection using YOLOv8 + Pose Estimation + LSTM

This project implements a real-time faint detection systemusing:

 1.YOLOv8 for person detection
 2.YOLO-Pose (Ultralytics) for pose landmark estimation
 3.LSTM model for temporal action classification (`faint` vs `normal`)

Deployed using a live CCTV or webcam feed, this system is ideal for monitoring individuals in isolated or high-risk environments — such as server rooms, labs, or industrial zones.


 Project Overview

 Input:

 Live video stream or `.mp4` dataset
 Detects humans, extracts their pose sequence over time

Output:

 Classifies the movement into:

   `Normal`
    `Faint`

 Pipeline:


[YOLOv8 Person Detector]
          ↓
[YOLO-Pose Landmark Estimator]
          ↓
[Sequence of Keypoints (30 frames)]
          ↓
[LSTM-based Classifier]
          ↓
[Real-time Label Display (Faint / Normal)]



#  How It Works

 1. Detection

* Uses YOLOv8 to detect people in each frame
* Crops region of interest for pose extraction

 2. Pose Estimation

YOLO-Pose predicts 17 or 33 body keypoints
Keypoints are normalized for scale and translation

 3. Temporal Modeling

A sequence of 30 frames (approx 1 second) is collected
 LSTM takes the sequence and classifies it as `faint` or `normal`

---

 📁 Directory Structure

```
├── dataset/
│   ├── normal/
│   └── faint/
├── X.npy / y.npy         ← Processed sequences
├── extract_poses.py      ← Keypoint sequence generation
├── train_lstm.py         ← LSTM training script
├── faint_detect.py       ← Real-time inference script
├── utils.py              ← Keypoint normalization and helper methods
├── model.pth             ← Trained LSTM model




  Training

Generate the sequence data from videos:

bash
python extract_poses.py


Train the LSTM:

bash
python train_lstm.py


 Real-Time Detection

Run on webcam:

bash
python faint_detect.py


Or test on a video file:

python
cap = cv2.VideoCapture("your_clip.mp4")



Features

 Real-time FPS (\~30fps on webcam)
 Softmax-based confidence scoring
 Auto-handling of missing keypoints
 Works with side and back poses using YOLO-Pose





