import torch
from PIL import Image
import numpy as np
import streamlit as st
from torchvision import transforms
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


class SkinTypePredictor(VideoTransformerBase):
    def __init__(self):
        super().__init__()

        # Define the URL from where to download the model
        self.model_url = 'https://github.com/basuru07/oil_face_detection/raw/main/model.pth'

        # Download the model file locally
        self.local_model_path = 'model.pth'
        torch.hub.download_url_to_file(self.model_url, self.local_model_path, progress=False)

        # Load the pre-trained model
        self.model = torch.load(self.local_model_path, map_location=torch.device('cpu'))
        self.model.eval()

        # Define index_label globally
        self.index_label = {0: "Dry Skin", 1: "Normal Skin",
                            2: "Oily Skin", 3: "High Oily Skin"}

    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image).unsqueeze(0)
        return image

    def transform(self, frame):
        # Perform face detection using OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            face = frame[y:y+h, x:x+w]

            # Preprocess the face image for inference
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = self.preprocess_image(face_pil)

            # Make predictions using the model
            with torch.no_grad():
                output = self.model(face_tensor)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()

            # Draw a rectangle around the face and put the prediction label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Skin Type: {}".format(self.index_label[prediction]), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame


def main():
    st.set_page_config(page_title="Skin detection", page_icon=":tdata", layout="wide")

    st.title("Live Skin Oilyness Detection Web Application")

    st.markdown(
        "<h3 style='color:#16056B'>Watch and Get Idea about your Skin</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/Rc4J0_Xg88w?si=x4gmIB6eq_cfkl2o" '
        f'title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
        f'encrypted-media; gyroscope; picture-in-picture; web-share" '
        f'referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='color:#16056B; margin-top: 20px'>Test your Skin Type</h3>",
        unsafe_allow_html=True
    )

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=SkinTypePredictor)

    if webrtc_ctx.video_transformer:
        st.markdown("<h5>Webcam Feed with Predictions:</h5>", unsafe_allow_html=True)
        stframe = st.empty()

        while True:
            try:
                frame = webrtc_ctx.video_transformer.get_frame()
                stframe.image(frame, channels="BGR", use_column_width=True)
            except AttributeError:
                pass


if __name__ == "__main__":
    main()
