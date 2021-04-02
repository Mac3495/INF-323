import numpy
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, decode_predictions

#creamos instancia
iv3 = InceptionV3()

#cargamos imagen
from google.colab import files
uploaded = files.upload()
x = image.img_to_array(image.load_img("dog.jpeg", target_size=(299, 299)))

#creamos dimensiones
x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
 
#analizar la imagen
keras.applications.inception_v3.preprocess_input(x)

y = iv3.predict(x)
print(decode_predictions(y))

#guardar prediccion
datos1 = decode_predictions(y)
print("Imagen clasificada como:")
print(datos1[0][0])

