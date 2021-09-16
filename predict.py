import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model


@st.cache
def Classify(img, model_name):

    if model_name == "Resnet50":
        model = "model_resnet50.h5"
    else:
        model = "model_resnet152v2.h5"

    model = load_model("model/{}".format(model))

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    normalized_image_array = image_array.astype(np.float32) / 255

    data[0] = normalized_image_array

    prediction = model.predict(data)
    return np.argmax(prediction)


classes_names = ['Diseased Cotton Leaf',
                 'Diseased Cotton Plant',
                 'Fresh Cotton Leaf',
                 'Fresh Cotton Plant']

st.title("Cotton Disease Classification")

uploaded_file = st.sidebar.file_uploader("Choose a cotton leaf or plant image ...", type="jpg")
model_name = st.sidebar.selectbox("Model Name", ("Restnet50", "Resnet152V2"))

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Cotton Leaf or Plant Image.', use_column_width=True)
    label = Classify(image, model_name)
    st.subheader("Result - " + classes_names[label])

else:
    st.text("Upload a cotton plant or leaf Image for image classification as disease or no-diease")
