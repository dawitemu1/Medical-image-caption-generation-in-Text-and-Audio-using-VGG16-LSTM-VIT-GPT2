import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from PIL import Image
import pyttsx3
import pickle
import torch

# Load your custom-trained model and tokenizer
@st.cache_resource
def load_custom_captioning_model():
    # Load the trained model
    model = VisionEncoderDecoderModel.from_pretrained("my_model.keras.h5", local_files_only=True)
    
    # Load the feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Load the tokenizer
    with open("caption.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    return model, feature_extractor, tokenizer

model, feature_extractor, tokenizer = load_custom_captioning_model()

# Function to generate captions from an image
def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    
    # Generate captions using the trained model
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4, early_stopping=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Function to convert text to WAV audio
def text_to_audio_wav(text, output_audio_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_audio_path)  # Save as WAV by default
    engine.runAndWait()

# Streamlit App
st.title("Image Caption to WAV Audio")

# Image Upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Variables to store the caption and audio path
caption = None
audio_path = "generated_audio.wav"

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to generate caption
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            try:
                caption = generate_caption(image)
                st.success("Caption generated successfully!")
                st.write(f"**Generated Caption:** {caption}")
            except Exception as e:
                st.error(f"Error generating caption: {e}")

    # Button to generate audio (only enabled if a caption exists)
    if caption:
        if st.button("Generate Audio"):
            with st.spinner("Converting caption to audio..."):
                try:
                    text_to_audio_wav(caption, audio_path)
                    st.success("WAV audio generated successfully!")

                    # Play the audio
                    with open(audio_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/wav")

                    # Option to download the audio
                    st.download_button(
                        label="Download WAV Audio",
                        data=open(audio_path, "rb").read(),
                        file_name="caption_audio.wav",
                        mime="audio/wav",
                    )
                except Exception as e:
                    st.error(f"Error converting text to audio: {e}")
