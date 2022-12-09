import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./flower_model_trained.hdf5')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Classificador de flores')

file = st.file_uploader("Carregue uma imagem: ", type=["jpg", "png"])


if file is None:
	st.text('Aguardando o upload ...')

else:
	slot = st.empty()
	slot.text('Rodando classificação ...')

	test_image = Image.open(file)

	st.image(test_image, caption="Imagem", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

	result = class_names[np.argmax(pred)]

	output = 'Tipo da flor: ' + result

	slot.text('Finalizado!')
	
	st.success(output)
	
	st.success(pred)

