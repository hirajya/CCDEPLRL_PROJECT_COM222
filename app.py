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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {BASE_DIR}")
    model_path = os.path.abspath(os.path.join(BASE_DIR, "models/baseline_model.h5"))
    print(f"Model path: {model_path}")

    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("Model not found. Please train the model first.")
        return None

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    # Ensure image is in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    img = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Ensure we have the right shape (224, 224, 3)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app
def main():
    st.title("Bruise Detection System")
    st.write("Upload an image to detect if it shows a bruise")
    
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
                
                # Get prediction and convert to native Python float
                prediction = model.predict(processed_img)
                probability = float(prediction[0][0])  # Convert from float32 to float
                
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
                # Convert probabilities to native Python float
                bruise_prob = float(probability)
                normal_prob = float(1 - probability)
                
                st.write("Bruise Probability:")
                st.progress(bruise_prob)
                st.write(f"{bruise_prob*100:.2f}%")
                
                st.write("Normal Skin Probability:")
                st.progress(normal_prob)
                st.write(f"{normal_prob*100:.2f}%")

if __name__ == "__main__":
    main()