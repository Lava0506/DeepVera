# 🕵️‍♀️ DeepVera: AI Deepfake Detection System
**Built by Team Pixel Pearl** | **Developer:** Lavanya Luhan

DeepVera is a full-stack, AI-powered deepfake detection web application. It processes user-uploaded videos, isolates human faces using advanced Computer Vision techniques, and runs them through a fine-tuned deep learning model to determine if the footage is authentic or artificially generated.

## ✨ Core Features
* **Automated Frame Extraction:** Slices raw `.mp4` video files into sequential, static image frames.
* **Intelligent Face Hunting:** Utilizes MTCNN to mathematically locate and tightly crop the subject's face, filtering out background noise.
* **Deep Learning Analysis:** Processes cropped faces through a fine-tuned ResNet18 neural network to calculate a Real vs. Fake authenticity score.
* **Web Interface (In Development):** A seamless Flask-based backend and HTML/JS frontend allowing users to upload videos directly from their browser.

## 🛠️ The Tech Stack
* **Computer Vision:** `OpenCV`, `MTCNN` (facenet-pytorch)
* **Deep Learning:** `PyTorch`, `ResNet18`
* **Web Server:** `Flask` (Python)
* **Frontend:** HTML5, CSS3, JavaScript

## 📁 Project Structure
```text
deepvera/
│
├── extract_frames.py         # Script to slice video into .jpg frames
├── crop_faces.py             # Script to isolate and crop faces via MTCNN
├── app.py                    # Flask backend server (Phase 3)
├── .gitignore                # Repository security shield
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
You will need **Python 3.8+** and **Git** installed on your machine.

### 2. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/DeepVera.git](https://github.com/YOUR_USERNAME/DeepVera.git)
cd DeepVera
```

### 3. Set up the Virtual Environment
It is highly recommended to run this project inside an isolated virtual environment.
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install opencv-python facenet-pytorch flask torch torchvision
```

### 5. ⚠️ CRITICAL: The "Missing" Files
To keep this GitHub repository clean and fast, heavy binary files are blocked via `.gitignore`. **Before running the project, you must manually add the following to your root `deepvera` folder:**
1. **The AI Brain:** Download the trained model weights (`deepvera_resnet18.pth`) and place it in the main folder.
2. **The Test Subject:** Add any `.mp4` video featuring a human face and name it `test_video.mp4`.

---

## 💻 How to Use the Project

Currently, the project is in **Phase 2** (Local Pipeline Testing). You can test the computer vision modules directly from your terminal.

**Step 1: Extract Video Frames**
Run the OpenCV script to chop your `test_video.mp4` into individual pictures.
```bash
python extract_frames.py
```
*Outcome: A new folder named `extracted_frames/` will appear containing the image sequence.*

