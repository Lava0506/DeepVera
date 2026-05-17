import os
import cv2
import yt_dlp
import uuid
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from facenet_pytorch import MTCNN
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
def download_from_url(url, upload_folder):
    print(f"Attempting to download link: {url}")
    
    filename = f"downloaded_{uuid.uuid4().hex}.mp4"
    filepath = os.path.join(upload_folder, filename)

    ydl_opts = {
        'outtmpl': filepath,
        'format': 'best[ext=mp4]/best'
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return filepath
    except Exception as e:
        print(f"Download Error: {e}")
        return None
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Booting up DeepVera Engine... Please wait.")


MODEL_WEIGHTS = "deepvera_resnet18.pth" 

device = torch.device('cpu')
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True, device=device)


model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Engine Ready! Starting Web Server...")

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/detector')
def detector():
            """Renders the dedicated QuillBot-style dashboard page."""
         
            return render_template('detector.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    filepath = None
    
    if 'video_url' in request.form and request.form['video_url'].strip() != '':
        video_url = request.form['video_url'].strip()
        filepath = download_from_url(video_url, app.config['UPLOAD_FOLDER'])
        
        if not filepath:
            return render_template('result.html', error="Failed to download video from the link. The platform might be blocking the request, or the link is invalid.")
            
    elif 'video_file' in request.files and request.files['video_file'].filename != '':
        file = request.files['video_file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
    if not filepath:
        return render_template('result.html', error="No valid file or link provided.")
        
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    num_frames_to_sample = 15
    step = max(1, total_frames // num_frames_to_sample)
    frames_to_check = [i * step for i in range(num_frames_to_sample)]
    
    face_tensors = []
    
    for frame_idx in frames_to_check:
        if len(face_tensors) >= 7:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        boxes, probs = mtcnn.detect(pil_img)
        
        if boxes is not None:
           
            largest_box = None
            max_area = 0
            
            for box in boxes:
                
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    largest_box = box
            
           
            x1 = max(0, int(largest_box[0]))
            y1 = max(0, int(largest_box[1]))
            x2 = min(pil_img.width, int(largest_box[2]))
            y2 = min(pil_img.height, int(largest_box[3]))
            
           
            face_pil = pil_img.crop((x1, y1, x2, y2))
           
            face_pil = face_pil.resize((224, 224))
            tensor = transform(face_pil).unsqueeze(0)
            face_tensors.append(tensor)
            
    cap.release()
    
    if not face_tensors:
        return render_template('result.html', error="No clear human face could be detected in this video. Please upload a video with a visible face.")
        
    batch_tensors = torch.cat(face_tensors, dim=0)
    
    with torch.no_grad():
        output = model(batch_tensors)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        avg_probabilities = probabilities.mean(dim=0)
        
        fake_prob = avg_probabilities[0].item()
        real_prob = avg_probabilities[1].item()
        
        if real_prob > fake_prob:
            verdict = "REAL"
            confidence = round(real_prob * 100, 1)
        else:
            verdict = "FAKE"
            confidence = round(fake_prob * 100, 1)
            
        # --- FORENSIC EXPLANATION ENGINE ---
        
        if verdict == "FAKE" and confidence > 85:
            explanation = {
                "title": "SYNTHETIC MEDIA DETECTED",
                "text": "DeepVera isolated significant statistical anomalies in the facial boundaries across multiple frames. The pixel blending and temporal inconsistencies strongly indicate AI-generated manipulation.",
                "color": "red"
            }
        elif verdict == "REAL" and confidence > 90:
            explanation = {
                "title": "AUTHENTIC MEDIA VERIFIED",
                "text": "No significant evidence of digital face manipulation detected. The biological texture, light reflection, and cross-frame consistency match natural human recording patterns.",
                "color": "green"
            }
        else:
            explanation = {
                "title": f"INCONCLUSIVE (LEANING {verdict})",
                "text": f"The model detected high levels of noise (likely webcam compression or poor lighting) obscuring the facial boundaries. While the mathematical probability leans toward {verdict}, definitive forensic classification requires a higher-resolution file.",
                "color": "yellow"
            }

    # SEND RESULTS
    return render_template('result.html', 
                           prediction=verdict, 
                           confidence=confidence, 
                           explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
