import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Precautions for each class
precautions = {
    "mature cataract": [
        "1. Consult an eye specialist for a detailed examination.",
        "2. Consider surgery if vision impairment affects daily activities.",
        "3. Avoid driving or operating machinery until vision is corrected.",
        "4. Use adequate lighting to prevent falls or accidents.",
        "5. Regularly monitor eye health and vision changes."
    ],
    "immature cataract": [
        "1. Schedule regular eye exams to monitor the condition.",
        "2. Protect your eyes from UV light by wearing sunglasses.",
        "3. Maintain a healthy diet rich in antioxidants (fruits and vegetables).",
        "4. Avoid eye strain by taking breaks during prolonged screen use.",
        "5. Discuss potential treatment options with an eye care professional."
    ],
    "healthy eye": [
        "1. Schedule regular eye check-ups to maintain eye health.",
        "2. Follow a balanced diet to support eye health.",
        "3. Wear protective eyewear during sports and high-risk activities.",
        "4. Manage screen time and take regular breaks to prevent eye strain.",
        "5. Stay hydrated and maintain a healthy lifestyle."
    ]
}

# Function to preprocess the image
def preprocess_image(image):
    # Resize and center crop
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Create the data array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

# Streamlit UI
st.title("Image Classification with Keras Model")
st.write("Upload an image to classify.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Preprocess the image
    data = preprocess_image(image)
    
    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove any trailing newline characters
    confidence_score = prediction[0][index]

    # Display the results
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Class:", class_name)
    st.write("Confidence Score:", confidence_score)

    # Display precautions
    if class_name == "mature cataract":
        st.write("**Precautions for Mature Cataract:**")
        for precaution in precautions["mature cataract"]:
            st.write(precaution)
    elif class_name == "immature cataract":
        st.write("**Precautions for Immature Cataract:**")
        for precaution in precautions["immature cataract"]:
            st.write(precaution)
    else:
        st.write("**Precautions for Healthy Eyes:**")
        for precaution in precautions["healthy eye"]:
            st.write(precaution)
