import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/bruise_detection_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app
st.title("Bruise Detection App")
st.write("Upload an image to detect if it contains a bruise.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    
    # Display the result
    if prediction > 0.5:
        st.write("### Prediction: Bruise detected! 🩹")
        st.write(f"Confidence: {prediction:.2f}")
    else:
        st.write("### Prediction: Normal skin detected! ✅")
        st.write(f"Confidence: {1 - prediction:.2f}")