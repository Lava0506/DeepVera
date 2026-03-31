# DeepVera 🕵️‍♀️🔍
**Team: Pixel Pearl** | **Developer: Lavanya Luhan**

DeepVera is an AI-powered deepfake detection web application. It processes user-uploaded videos, isolates human faces using Computer Vision, and runs them through a fine-tuned deep learning model to determine if the footage is authentic or artificially generated.

## ⚙️ The Tech Stack
This project uses a multi-stage pipeline to process and grade video content:
* **Frontend:** HTML / CSS / JavaScript (User Interface)
* **Backend:** Flask (Python Web Server)
* **Computer Vision (Image Processing):** * `OpenCV` - Slices uploaded `.mp4` videos into individual static frames.
    * `MTCNN` (facenet-pytorch) - Scans frames to locate and tightly crop the subject's face, removing background noise.
* **Artificial Intelligence (Deep Learning):**
    * `PyTorch` - Core machine learning framework.
    * `ResNet18` - Pre-trained convolutional neural network fine-tuned specifically for deepfake detection (Binary Classification: Real vs. Fake).

## 📁 Project Structure
```text
deepvera/
│
├── extract_frames.py         # OpenCV script to slice video into .jpg frames
├── crop_faces.py             # MTCNN script to isolate and crop faces
├── app.py                    # Flask backend server (In Progress)
├── .gitignore                # Shields repo from heavy files
└── README.md
