import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from copy import deepcopy

# Define the URL from where to download the model
model_url = 'https://github.com/basuru07/oil_face_detection/raw/main/model.pth'

# Download the model file locally
local_model_path = 'model.pth'
torch.hub.download_url_to_file(model_url, local_model_path, progress=False)

# Load your pre-trained model from the local file
best_model = torch.load(local_model_path, map_location=torch.device('cpu'))
best_model.eval()

# Define index_label globally
index_label = {0: "Dry Skin", 1: "Normal Skin",
               2: "Oily Skin", 3: "High Oily Skin"}

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()

    def transform(self, frame):
        img = Image.fromarray(frame)
        original_img = deepcopy(img)
        img = preprocess_image(img)  # Corrected line
        img = img.view(1, 3, 224, 224)
        with torch.no_grad():
            out = best_model(img)
            skin_type = index_label[out.argmax(1).item()]
        return np.array(original_img), skin_type

def main():
    st.set_page_config(page_title="Skin detection", page_icon=":tdata", layout="wide")

    st.title("Live Skin Oilyness Detection Web Application")

    st.markdown(
        "<h3 style='color:#16056B'>Predict your Skin Type</h3>",
        unsafe_allow_html=True
    )

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if webrtc_ctx.video_transformer:
        selected_skin_type = webrtc_ctx.video_transformer.skin_type
        st.write(f"Predicted Skin Type: {selected_skin_type}")

if __name__ == "__main__":
    main()
