import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Bruise Detection System",
    layout="wide"
)

# Model loading function
@st.cache_resource
def load_model():
    model_path = os.path.join('..', 'models', 'bruise_detection_model.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("Model not found. Please train the model first.")
        return None

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    # Resize image
    img = image.resize(target_size)
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app
def main():
    st.title("Bruise vs Normal Skin Detection")
    st.write("Upload an image to detect if it shows a bruise or normal skin")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Load model
        model = load_model()
        
        if model:
            # Make prediction
            with st.spinner("Analyzing image..."):
                # Preprocess the image
                processed_img = preprocess_image(image)
                
                # Get prediction
                prediction = model.predict(processed_img)
                probability = prediction[0][0]
                
                # Display results
                st.subheader("Detection Results:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    result = "Bruise" if probability > 0.5 else "Normal Skin"
                    st.metric("Prediction", result)
                
                with col2:
                    confidence = probability if probability > 0.5 else 1 - probability
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Display probability distribution
                st.subheader("Probability Distribution:")
                
                # Create two bars for bruise and normal probabilities
                bruise_prob = probability
                normal_prob = 1 - probability
                
                st.write("Bruise Probability:")
                st.progress(bruise_prob)
                st.write(f"{bruise_prob*100:.2f}%")
                
                st.write("Normal Skin Probability:")
                st.progress(normal_prob)
                st.write(f"{normal_prob*100:.2f}%")

if __name__ == "__main__":
    main()