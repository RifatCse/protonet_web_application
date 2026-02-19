import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. DEFINE ARCHITECTURE (Must match your notebook exactly)
class PrototypicalNetworks(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, support_images, support_labels, query_images):
        z_support = self.feature_extractor.forward(support_images)
        z_query = self.feature_extractor.forward(query_images)
        n_way = len(torch.unique(support_labels))
        
        # Calculate prototypes for each class
        z_proto = torch.cat([
            z_support[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ])
        
        # Calculate distances (Euclidean)
        dists = torch.cdist(z_query, z_proto)
        return -dists # Return negative distance so highest value = best class

# 2. HELPER TO LOAD MODEL
@st.cache_resource
def load_model():
    backbone = models.mobilenet_v3_large(weights=None)
    backbone.classifier = nn.Sequential(
        nn.Linear(backbone.classifier[0].in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128)
    )
    model = PrototypicalNetworks(backbone)
    # Load weights
    model.load_state_dict(torch.load("mobilenet_proto_model.pth", map_location="cpu"))
    model.eval()
    return model

# 3. PREPROCESSING (Using your exact notebook logic)
image_size = 244
transform = transforms.Compose([
    transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])

# STREAMLIT UI
st.title("Plant Disease Detector (Few-Shot)")
st.sidebar.header("Configuration")

# Support Set Management
st.sidebar.write("### 1. Upload Support Set (5 images per class)")
# In a real app, you might pre-load these from a folder. 
# For this example, we define the classes from your notebook:
CLASS_NAMES = ['Healthy', 'Mosaic', 'Yellow'] #

# UI Logic
uploaded_query = st.file_uploader("Upload leaf to test", type=['jpg','png','jpeg'])

if uploaded_query:
    # We need a support set to make a prediction. 
    # For a production app, it is best to have these images stored in a 'support/' folder
    support_path = "support_images/" # Folder with subfolders 'Healthy', 'Mosaic', etc.
    
    if os.path.exists(support_path):
        model = load_model()
        
        # Prepare Support Tensors
        support_imgs = []
        support_labs = []
        
        for idx, cls in enumerate(CLASS_NAMES):
            cls_folder = os.path.join(support_path, cls)
            # Take first 5 images found in folder
            files = os.listdir(cls_folder)[:5]
            for f in files:
                img = Image.open(os.path.join(cls_folder, f)).convert('RGB')
                support_imgs.append(transform(img))
                support_labs.append(idx)
        
        support_tensor = torch.stack(support_imgs)
        support_labels = torch.tensor(support_labs)
        query_tensor = transform(Image.open(uploaded_query).convert('RGB')).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(support_tensor, support_labels, query_tensor)
            prediction = torch.argmax(logits, dim=1).item()

        if CLASS_NAMES[prediction]: 
            st.success(f"Prediction: **{CLASS_NAMES[prediction]}**")
        else:
            print("Loading...")
            
        st.image(uploaded_query, caption="Query Image", width=300)
    else:
        st.error("Please create a folder named 'support_images' with subfolders for each class containing 5 images each.")