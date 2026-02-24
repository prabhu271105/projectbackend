import io
import json
import urllib.request
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

import torch
from torchvision import models, transforms

app = FastAPI()

# --- 1. LOAD MODEL ---
print("Loading PyTorch MobileNetV3 model...")
# We use the 'small' version with default ImageNet weights
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)
model.eval() # Set model to evaluation mode (crucial for inference)
print("Model loaded successfully!")

# --- 2. IMAGE PREPROCESSING ---
# PyTorch models require specific transformations. 
# The weights object actually contains the exact transforms needed!
preprocess = weights.transforms()

# --- 3. LOAD IMAGENET LABELS ---
# PyTorch outputs raw class indices (0-999). We need the human-readable labels.
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
try:
    with urllib.request.urlopen(LABELS_URL) as url:
        imagenet_labels = json.loads(url.read().decode())
    print("Labels loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load labels: {e}")
    imagenet_labels = ["Unknown"] * 1000

@app.get("/")
def root():
    return {"message": "PyTorch MobileNetV3 API is ready."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Basic validation
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 1. Read and open the image via Pillow
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 2. Preprocess the image
        # This applies resizing, cropping, and normalization specific to MobileNetV3
        input_tensor = preprocess(img)
        
        # 3. Create a mini-batch
        # PyTorch models expect batches of images. 
        # We turn our single image of shape (C, H, W) into a batch of 1: (1, C, H, W)
        input_batch = input_tensor.unsqueeze(0) 

        # 4. Run Inference
        # torch.no_grad() disables gradient calculation, saving memory and speeding up prediction
        with torch.no_grad():
            output = model(input_batch)

        # 5. Calculate Probabilities
        # The model outputs raw "logits". We use Softmax to convert these into percentages (0.0 to 1.0)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 6. Get Top 3 Predictions
        top3_prob, top3_catid = torch.topk(probabilities, 3)

        # 7. Format Results for the Mobile App
        results = []
        for i in range(top3_prob.size(0)):
            # .item() extracts the standard Python number from the PyTorch Tensor
            score = top3_prob[i].item()
            category_id = top3_catid[i].item()
            label = imagenet_labels[category_id]
            
            results.append({
                "label": label,
                "confidence": score
            })

        return {"predictions": results}

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error processing image on the server.")