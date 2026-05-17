import os
from facenet_pytorch import MTCNN
from PIL import Image



# 1. Setup the AI
mtcnn = MTCNN(
    image_size=224,       
    margin=20,            
    keep_all=False,       
    select_largest=True,  
    device='cpu'          
)

input_folder = "extracted_frames"   
output_folder = "cropped_faces"     
os.makedirs(output_folder, exist_ok=True)

# 2. Process the images
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        
        # Read the image
        img = Image.open(img_path).convert('RGB')
        save_path = os.path.join(output_folder, f"cropped_{filename}")
        
        # Crop and save
        face_tensor = mtcnn(img, save_path=save_path)
        
        if face_tensor is not None:
            print(f"Face cropped: {filename}")
        else:
            print(f"No face detected in: {filename}")

print("Done! Check your cropped_faces folder.")