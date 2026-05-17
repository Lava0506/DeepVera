import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image



# 1. Setup the Paths
input_folder = "cropped_faces"
model_weights_path = "deepvera_resnet18.pth" 

# 2. Rebuild the AI Architecture (The Surgery)
# We have to build the exact same body that the brain used during training
model = models.resnet18(weights=None)  # Start with an empty ResNet18
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)      # Change the output layer to 2 choices (Real/Fake)

# Load your custom brain into the empty body
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()  # Lock the weights. Tell the AI it is time to take the test, not study.

# 3. Setup the Math Translator (Image to Tensor)
# ResNet strictly requires this exact mathematical normalization to work properly
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Predict the Faces
count1 = 0  # Real
count2 = 0  # Fake
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        
        # Open image and translate to Tensor
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0) # Adds the Batch Dimension: [1, 3, 224, 224]
        
        # Pass it into the brain
        with torch.no_grad(): # Don't waste memory calculating gradients
            output = model(img_tensor)
           
            
            # Figure out which of the 2 choices has the highest score
            _, prediction = torch.max(output, 1)
            
            # NOTE: If your output says Fake when it should be Real, swap these two!
            # Usually: 0 = Fake, 1 = Real. It depends on how your dataset folders were sorted.
            if prediction.item() == 1:
                result = "🟢 REAL" 
                count1+=1
            else:
                result = "🔴 FAKE"
                count2+=1
            
            print(f"File: {filename} --> Prediction: {result}")
            
print(f"\nTotal Real: {count1}, Total Fake: {count2}")
percentage_real = (count1 / (count1 + count2)) * 100 if (count1 + count2) > 0 else 0
print(f"Percentage Real: {percentage_real:.2f}%")
print("\n--- doneeee💕🧚🏼---")