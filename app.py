import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3_custom
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_vgg16

# Load the trained models
model_path_vgg16 = "Vgg16.h5"
model_vgg16 = load_model(model_path_vgg16, compile=False)

model_path_vgg19 = "vgg19.h5"
model_vgg19 = load_model(model_path_vgg19, compile=False)

model_path_inceptionv3 = "inceptionv3.h5"
model_inceptionv3 = load_model(model_path_inceptionv3, compile=False)

# Define class labels (replace with your own class labels)
class_labels = ['Bengin', 'adenocarcinoma', 'largecell', 'normal', 'squamous']

# Rename the InceptionV3 preprocessing function


def preprocess_input_inceptionv3_custom(image):
    return preprocess_input_inceptionv3(image)

# Function to preprocess the image for VGG16


def preprocess_image_vgg16(image):
    img = image.resize((299, 299))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input_vgg16(img_array)

# Function to preprocess the image for VGG19


def preprocess_image_vgg19(image):
    img = image.resize((299, 299))  # Change the size to (299, 299)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input_vgg19(img_array)

# Function to preprocess the image for InceptionV3


def preprocess_image_inceptionv3(image):
    img = image.resize((299, 299))  # Change the size to (299, 299)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input_inceptionv3_custom(img_array)

# Function to make predictions for all models


def classify_images(image, model_vgg16, model_vgg19, model_inceptionv3):
    preprocessed_image_vgg16 = preprocess_image_vgg16(image)
    preprocessed_image_vgg19 = preprocess_image_vgg19(image)
    preprocessed_image_inceptionv3 = preprocess_image_inceptionv3(image)

    prediction_vgg16 = model_vgg16.predict(preprocessed_image_vgg16)
    predicted_class_vgg16 = np.argmax(prediction_vgg16)

    prediction_vgg19 = model_vgg19.predict(preprocessed_image_vgg19)
    predicted_class_vgg19 = np.argmax(prediction_vgg19)

    prediction_inceptionv3 = model_inceptionv3.predict(
        preprocessed_image_inceptionv3)
    predicted_class_inceptionv3 = np.argmax(prediction_inceptionv3)

    return (
        class_labels[predicted_class_vgg16],
        class_labels[predicted_class_vgg19],
        class_labels[predicted_class_inceptionv3]
    )

# Streamlit app


def main():
    st.title('Image Classification with VGG16, VGG19, and InceptionV3')
    st.write(
        'Upload an image, and the models will classify it into one of the classes.')

    uploaded_image = st.file_uploader(
        'Choose an image...', type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify'):
            with st.spinner('Classifying...'):
                prediction_vgg16, prediction_vgg19, prediction_inceptionv3 = classify_images(
                    image, model_vgg16, model_vgg19, model_inceptionv3)
                st.success(f'VGG16 Prediction: {prediction_vgg16}')
                st.success(f'VGG19 Prediction: {prediction_vgg19}')
                st.success(f'InceptionV3 Prediction: {prediction_inceptionv3}')


if __name__ == '__main__':
    main()
