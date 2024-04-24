import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Load the trained model
# Replace with the actual path to your VGG16 model file
model_path = "C:\\Users\\vigne\\OneDrive\\Desktop\\plrdatasets\\afterreportmodel\\vgg16.h5"
model = load_model(model_path, compile=False)

# Define class labels (replace with your own class labels)
class_labels = ['Bengin', 'adenocarcinoma', 'largecell', 'normal', 'squamous']

# Function to preprocess the image


def preprocess_image(image):
    img = image.resize((299, 299))  # VGG16 requires input size (224, 224)
    img_array = np.array(img)

    # Ensure the image has 3 color channels (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to make predictions


def classify_image(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

# Streamlit app


def main():
    st.title('Image Classification with VGG16')
    st.write('Upload an image and the model will classify it into one of the classes.')

    uploaded_image = st.file_uploader(
        'Choose an image...', type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify'):
            with st.spinner('Classifying...'):
                prediction = classify_image(image, model)
                st.success(f'Prediction: {prediction}')


if __name__ == '__main__':
    main()
