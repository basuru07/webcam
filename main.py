from streamlit_webrtc import webrtc_streamer
import av
import torch
from PIL import Image
import numpy as np
import streamlit as st
import cv2
import requests
from torchvision import transforms
from io import BytesIO
import base64
from copy import deepcopy

# Load the model
model_url = 'https://github.com/basuru07/oil_face_detection/raw/main/model.pth'
local_model_path = 'model.pth'
response = requests.get(model_url)
with open(local_model_path, 'wb') as f:
    f.write(response.content)
best_model = torch.load(local_model_path, map_location=torch.device('cpu'))
best_model.eval()

# Define index_label globally
index_label = {0: "Dry Skin", 1: "Normal Skin", 2: "Oily Skin", 3: "High Oily Skin"}

# Define functions
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

def predict_skin_type_and_oiliness_level(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
    original_img = deepcopy(img)
    img = preprocess_image(img)
    with torch.no_grad():
        out = best_model(img)
        skin_type = index_label[out.argmax(1).item()]
        oiliness_level = np.argmax(out.cpu().numpy()) + 1
        return original_img, skin_type, oiliness_level

# Streamlit App
st.set_page_config(page_title="Skin detection", page_icon=":tdata", layout="wide")
st.markdown("<h1 style='color:#16056B;'>Live Skin Oiliness Detection Web Application</h1>", unsafe_allow_html=True)

# Webcam callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    original_img, skin_type, oiliness_level = predict_skin_type_and_oiliness_level(img)
    # Display results
    st.image(original_img, caption=f"Predicted Skin Type: {skin_type}, Oiliness Level: {oiliness_level}", use_column_width=True)

# Start the webcam
webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
