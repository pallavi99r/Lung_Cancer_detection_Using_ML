import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from tensorflow.keras.preprocessing import image as keras_image

# Define class labels (replace with your own class labels)
class_labels = ['Bengin', 'adenocarcinoma', 'largecell', 'normal', 'squamous']

# Function to preprocess the image for InceptionV3 model


def preprocess_inception_v3_image(image):
    img = image.resize((299, 299))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_inception_v3(img_array)

# Function to preprocess the image for VGG16 model


def preprocess_vgg16_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_vgg16(img_array)

# Function to preprocess the image for VGG19 model


def preprocess_vgg19_image(image):
    img = image.resize((299, 299))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_vgg19(img_array)

# Function to make predictions for different models


def classify_image(image, model, preprocess_func):
    preprocessed_image = preprocess_func(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

# Streamlit app


def main():
    st.title('Image Classification App')
    st.write(
        'Upload an image and choose a model to classify it into one of the classes.')

    uploaded_image = st.file_uploader(
        'Choose an image...', type=['jpg', 'png', 'jpeg'])
    model_choice = st.selectbox(
        'Select a Model', ('InceptionV3', 'VGG16', 'VGG19'))

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        model_path = ""

        if model_choice == 'InceptionV3':
            model_path = "inceptionv3.h5"
            preprocess_func = preprocess_inception_v3
        elif model_choice == 'VGG16':
            model_path = "vgg16.h5"
            preprocess_func = preprocess_vgg16
        elif model_choice == 'VGG19':
            model_path = "vgg19.h5"
            preprocess_func = preprocess_vgg19

        model = load_model(model_path, compile=False)

        if st.button('Classify'):
            with st.spinner('Classifying...'):
                prediction = classify_image(image, model, preprocess_func)
                st.success(f'Prediction: {prediction}')


if __name__ == '__main__':
    main()
