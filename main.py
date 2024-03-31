import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from copy import deepcopy
from torchvision import transforms
import base64
from io import BytesIO
import cv2
import torch
import requests




def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image


# Define index_label globally
index_label = {0: "Dry Skin", 1: "Normal Skin",
               2: "Oily Skin", 3: "High Oily Skin"}


def predict_skin_type_and_oiliness_level(image_path):
    img = Image.open(image_path).convert("RGB")
    original_img = deepcopy(img)
    img = preprocess_image(img)  # Corrected line
    img = img.view(1, 3, 224, 224)
    best_model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()

        out = best_model(img)
        skin_type = index_label[out.argmax(1).item()]
        oiliness_level = get_oiliness_level(out.cpu().numpy())

        return original_img, skin_type, oiliness_level


def get_oiliness_level(predictions):
    oiliness_index = np.argmax(predictions)
    # Assuming the predictions are in the order of dry, normal, oily
    if oiliness_index == 0:
        return 1  # Dry or very low
    elif oiliness_index == 1:
        return 2  # Low
    elif oiliness_index == 2:
        return 3  # Medium
    else:
        return 4  # Very high


def save_to_firestore(skin_type, oiliness_level):
    doc_ref = db.collection("oiltype").document()
    doc_ref.set({
        "Skin Type": skin_type,
        "Oiliness Level": oiliness_level
    })


def main():
    # Content of the Streamlit app
    st.set_page_config(page_title="Skin detection",
                       page_icon=":tdata", layout="wide")
    st.markdown(
        "<h1 style='color:#16056B;'>Live Skin Oilyness Detection Web Application</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='color:#16056B'>Introduction of our Application</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: justify;'>The application utilizes image processing techniques and ResNet architecture to detect and classify skin oiliness levels. Leveraging deep learning, it accurately identifies skin types as dry, normal, oily, or highly oily based on visual cues. By analyzing facial features, it provides valuable insights for skincare routines tailored to individual needs, aiding in the management of skin health. This innovative tool offers a non-invasive and efficient method for users to assess and understand their skin's condition.The application is an innovative oilyness detection app utilizing image processing and ResNet technology. Leveraging advanced deep learning techniques, it accurately analyzes skin types – dry, normal, oily, and high oily – from facial images. By harnessing ResNet's robust architecture, it overcomes complexities in image analysis and delivers reliable results. Users benefit from a user-friendly interface that swiftly processes webcam feed, providing real-time assessments of skin condition.</p>",
        unsafe_allow_html=True
    )
    st.button("View more", type="primary")
    st.markdown(
        "<h3 style='color:#16056B'>Watch and Get Idea about your Skin</h3>",
        unsafe_allow_html=True
    )
    with st.container():
        left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(
            f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/Rc4J0_Xg88w?si=x4gmIB6eq_cfkl2o" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>',
            unsafe_allow_html=True
        )
    with right_column:
        st.markdown(
            f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/onkB0EN0zas?si=_8TMmmDbfwFZu4oX" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>',
            unsafe_allow_html=True
        )
    st.markdown(
        "<h3 style='color:#16056B; margin-top: 30px'>Test your Skin Type</h3>",
        unsafe_allow_html=True
    )
    with st.container():
        left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(
            "<p style='text-align: justify;'>Test Your Skin Type offers a cutting-edge solution for personalized skincare assessment. With its intuitive interface and advanced image processing capabilities, this application empowers users to determine their skin type effectively. Leveraging sophisticated ResNet technology, it accurately detects four distinct skin types: dry, normal, oily, and high oily. The application  seamless integration of deep learning algorithms ensures precise analysis, enabling users to make informed decisions about their skincare routines. Upon initiating the test, users simply need to provide a facial image via webcam feed, and the application swiftly delivers real-time results. Coupled with its elegant design and user-friendly experience, revolutionizes skincare diagnostics, providing individuals with invaluable insights into their skin's unique needs. Whether seeking hydration solutions or oil-control strategies, this application serves as a trusted companion, promoting healthier skin and bolstering self-confidence.</p>",
            unsafe_allow_html=True
        )
    with right_column:
        st.markdown(
            f'<img src="https://img.freepik.com/premium-vector/oily-dry-skin-different-human-skin-types-conditions-diagrammatic-sectional-view-skin_137876-131.jpg?w=1060" width="100%" height="300">',
            unsafe_allow_html=True
        )

    st.markdown(
        "<h3 style='color:#16056B; margin-top: 20px'>Predict your Skin Type</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: justify;'>Click the button below to use the webcam or upload your face image and get efficient skin type results. Our application utilizes advanced image processing to analyze facial features accurately. It swiftly detects skin types—dry, normal, oily, or highly oily—providing users with valuable insights into their skincare needs.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h5 style='color:#16056B'>Upload your Image and Get Prediction</h5>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Image upload", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        # Predict skin type and oiliness level
        original_img, skin_type, oiliness_level = predict_skin_type_and_oiliness_level(
            uploaded_file)
        # Display the original image with prediction results
        st.markdown("<h5 style='color:#FF4B4B;'><span>Result :- Predicted Skin Type is</span> {} <span> and Oilyness Level is</span> {}</h5>".format(
            '<span style="color:#FF4B4B;">{}</span>'.format(skin_type), '<span style="color:#FF4B4B;">{}</span>'.format(oiliness_level)), unsafe_allow_html=True)

        with st.container():
            left_column, right_column = st.columns([1, 2])
        with left_column:
            st.markdown(
                f'<div style="display: flex; justify-content:left;">'
                f'<img src="data:image/png;base64,{image_to_base64(original_img)}" style="width:400px;height:380px;margin-bottom:20px">'
                f'</div>',
                unsafe_allow_html=True
            )
        with right_column:
            # Check if the skin type is oily and provide treatment advice
            if skin_type == "Oily Skin":
                st.markdown("<h4 style='color:#16056B;'><span>Treatment for your {}</span></h4>".format(
                    '<span style="color:#16056B;">{}</span>'.format(skin_type)), unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: justify;'>For individuals grappling with oily skin, establishing an effective skincare regimen is paramount to manage excess oil production and foster a balanced complexion. It all starts with a diligent cleansing routine employing a gentle yet potent foaming cleanser to effectively eliminate impurities and excess oil without compromising the skin's natural moisture barrier. Follow this up with a non-alcoholic toner infused with ingredients such as witch hazel or tea tree oil, which helps rebalance the skin's pH levels and tighten pores, thus reducing the appearance of oiliness. Integration of lightweight, oil-free moisturizers formulated with hydrating agents like hyaluronic acid or glycerin is crucial to provide adequate hydration without exacerbating shine. Regular exfoliation using products containing salicylic acid or glycolic acid aids in unclogging pores, removing dead skin cells, and minimizing the appearance of enlarged pores.<br> Incorporating a clay mask into the weekly skincare routine proves beneficial in absorbing excess oil, purifying pores, and mattifying the skin. Throughout the day, utilize oil-absorbing sheets to control shine without disrupting makeup. Additionally, applying a broad-spectrum sunscreen daily is indispensable to protect against UV damage and premature aging. By adhering to these steps meticulously, individuals with oily skin can achieve a balanced, healthy complexion, minimizing the challenges posed by excessive oil production and embracing the confidence that comes with radiant, well-nourished skin.</p>",
                    unsafe_allow_html=True
                )
            if skin_type == "Dry Skin":
                st.markdown("<h4 style='color:#16056B;'><span>Treatment for your {}</span></h4>".format(
                    '<span style="color:#16056B;">{}</span>'.format(skin_type)), unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: justify;'>Dry skin requires a meticulous and nurturing skincare regimen to combat its challenges and restore a healthy, radiant complexion. Initiate your routine with a gentle, creamy cleanser, adept at purifying the skin without depleting its natural oils, thus laying a foundation of hydration and vitality. Follow this with a hydrating toner enriched with ingredients like hyaluronic acid or rose water, infusing your skin with much-needed moisture and soothing relief. Next, embrace the indulgence of a rich, emollient moisturizer, fortified with shea butter, ceramides, or squalane, to deeply hydrate and fortify the skin barrier, shielding against environmental aggressors. Regular exfoliation is key, employing either a gentle scrub or chemical exfoliant to slough off dead skin cells and stimulate cell turnover, unveiling a smoother, more luminous complexion.<br> Consider integrating a hydrating serum or facial oil for an added moisture boost, replenishing and nourishing the skin from within. Supplement your routine with a weekly hydrating mask, providing intense moisture infusion and enhancing skin elasticity for a revitalized appearance. Finally, shield your skin from environmental stressors by diligently applying a broad-spectrum sunscreen daily, safeguarding its newfound luminosity and resilience. This comprehensive approach ensures that dry skin is not just managed but transformed into a canvas of radiant vitality and enduring beauty, elevating your skincare routine to a realm of indulgent self-care and revitalization.</p>",
                    unsafe_allow_html=True
                )
            if skin_type == "Normal Skin":
                st.markdown("<h4 style='color:#16056B;'><span>Treatment for your {}</span></h4>".format(
                    '<span style="color:#16056B;">{}</span>'.format(skin_type)), unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: justify;'>Congratulations on possessing normal skin, a blessing that requires minimal specialized care, affording you the luxury of a naturally radiant and healthy complexion. Normal skin is characterized by its exquisite balance, boasting few imperfections, a smooth texture, and neither an excess of oiliness nor dryness. Maintaining its vitality is straightforward with a streamlined skincare regimen. Commence your routine with a gentle yet thorough cleansing to rid the skin of impurities, followed by nourishing moisturization to preserve its suppleness. Daily safeguarding against UV radiation with sunscreen is paramount to shield your skin from damage and preserve its innate allure.<br> Furthermore, integrate supplementary skincare rituals like weekly exfoliation to invigorate cellular turnover and occasional mask treatments to elevate hydration levels and provide additional nourishment. Embrace the harmonious equilibrium of your skin and revel in its innate splendor. Celebrate its resilience by adopting a holistic approach to skincare, ensuring it remains a testament to your overall well-being and self-care journey. With normal skin, the world is your oyster, and your skincare routine serves as a delightful indulgence rather than a necessity, allowing you to bask in the beauty of simplicity and the joy of effortless radiance. So, continue to cherish and nurture your skin, for it is a canvas of beauty and a reflection of your inner vitality and vibrancy.</p>",
                    unsafe_allow_html=True
                )

    st.markdown(
        "<h5 style='color:#16056B'>Open your Webcamera and Get Prediction</h5>",
        unsafe_allow_html=True
    )

    # Create a button to start webcam
    if st.button("Open Webcam", type="primary"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        st.markdown(
            "<h3 style='color:#16056B; margin-top:30px'>Watch the videos and get idea about skin Treatment</h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: justify;'>After watching the videos on skincare treatment, it's evident that a holistic approach is essential. From gentle cleansing to targeted treatments, addressing specific skin concerns is paramount. Incorporating hydrating serums, exfoliants, and SPF protection is crucial. Understanding individual skin types and concerns guides the selection of suitable products. Consistency and patience are key in achieving desired results. Emphasizing hydration, protection, and nourishment ensures skin health and vitality. Consulting a dermatologist for personalized advice can optimize skincare routines. By prioritizing proper skincare practices, one can attain a radiant complexion and boost confidence.</p>",
            unsafe_allow_html=True
        )
        with st.container():
            left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.markdown(
                f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/gQI7hYL8hpE?si=8vPTujAi18ositsR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )

        with middle_column:
            st.markdown(
                f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/5k93deKAkxw?si=tcMupJ-6YGRKwx5u" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )

        with right_column:
            st.markdown(
                f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/0Y7H-GMJ3Dg?si=zSd45MIBAqKGdURE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )

        # Loop to capture frames from the webcam
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform face detection using your preferred method (e.g., OpenCV's face detector)
            # Here's a simple example using Haar Cascade classifier
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # For each detected face, make predictions
            for (x, y, w, h) in faces:
                # Extract the face region
                face = frame[y:y+h, x:x+w]

                # Preprocess the face image
                face_pil = Image.fromarray(
                    cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = preprocess_image(face_pil)

                # Make predictions using the model
                with torch.no_grad():
                    output = best_model(face_tensor)
                    _, predicted = torch.max(output, 1)
                    prediction = predicted.item()

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

                # Draw a rectangle around the face and put the prediction label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Skin Type: {}".format(
                    skin_type), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Oilyness Level: {}".format(
                    level), (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Convert the frame to a format that Streamlit can display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

            # Check if 'q' is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam
        cap.release()


if __name__ == "__main__":
    main()
