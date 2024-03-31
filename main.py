import torch
from PIL import Image
import numpy as np
import streamlit as st
from torchvision import transforms
import cv2
import requests
from streamlit_webrtc import webrtc_streamer

# Define the URL from where to download the model
model_url = 'https://github.com/basuru07/oil_face_detection/raw/main/model.pth'

# Download the model file locally
local_model_path = 'model.pth'
with open(local_model_path, 'wb') as f:
    response = requests.get(model_url)
    f.write(response.content)

# Load your pre-trained model from the local file
best_model = torch.load(local_model_path, map_location=torch.device('cpu'))
best_model.eval()

# Define a function to preprocess the image for inference
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Define index_label globally
index_label = {0: "Dry Skin", 1: "Normal Skin", 2: "Oily Skin", 3: "High Oily Skin"}

# Define the Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title="Skin detection", page_icon=":tdata", layout="wide")

    # Function to preprocess the image from webcam and make predictions
    def predict_skin_type_and_oiliness_level(frame):
        # Convert frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")

        # Preprocess the image
        img = preprocess_image(img)

        # Make predictions using the model
        with torch.no_grad():
            output = best_model(img)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()

        # Determine skin type and oiliness level
        if prediction == 0:
            skin_type = "Dry Skin"
            level = "Level 1"
        elif prediction == 1:
            skin_type = "Normal Skin"
            level = "Level 2"
        elif prediction == 2:
            skin_type = "Oily Skin"
            level = "Level 3"
        else:
            skin_type = "High Skin"
            level = "Level 4"

        return skin_type, level

    # Create the Streamlit app content
    st.title("Skin Oiliness Detection")
    st.markdown("Open your webcam to detect skin oiliness in real-time.")

    # Function to process webcam frames
    def process_frame(frame):
        try:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make predictions on the frame
            skin_type, level = predict_skin_type_and_oiliness_level(frame)

            # Draw the prediction results on the frame
            cv2.putText(frame, f"Skin Type: {skin_type}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Oiliness Level: {level}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        return frame

    # Start the webcam and display predictions
    webrtc_streamer(key="example", video_processor_factory=process_frame)

if __name__ == "__main__":
    main()
