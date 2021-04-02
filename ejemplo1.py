from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Vemos el dataset
ids_imgs = np.random.randint(0,x_train.shape[0],16)
for i in range(len(ids_imgs)):
	img = x_train[ids_imgs[i],:,:]
	plt.subplot(4,4,i+1)
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	plt.title(y_train[ids_imgs[i]])
plt.suptitle('Números del dataset')
plt.show()

# Pre procesamos las imagenes a un vector 28x28 = 784 valores
X_train = np.reshape( x_train, (x_train.shape[0],x_train.shape[1]*x_train.shape[2]) )
X_test = np.reshape( x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[2]) )

# Normalizamos los valores entre 0-1
X_train = X_train/255.0
X_test = X_test/255.0

# Convertimos y_train y y_test a "one-hot"
nclasses = 10
Y_train = np_utils.to_categorical(y_train,nclasses) #[0,0,0,1,0,0,0,0,0,0,0,0]
Y_test = np_utils.to_categorical(y_test,nclasses)

# Creamos el modelo:
# Capa de entrada: 784 neuronas
# Capa oculta: 15 neuronas, Relu
# Capa de salida: Softmax 

np.random.seed(1)
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]

modelo = Sequential()
modelo.add( Dense(15, input_dim=input_dim, activation='relu'))
modelo.add( Dense(output_dim, activation='softmax'))
print(modelo.summary())

sgd = SGD(lr=0.2)
modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

num_epochs = 50
batch_size = 1024
historia = modelo.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=2)

#Mostramos el error y la exactitud del modelo
plt.subplot(1,2,1)
plt.plot(historia.history['loss'])
plt.title('Pérdida / iteraciones')
plt.ylabel('Pérdida')
plt.xlabel('Iteraciones')

plt.subplot(1,2,2)
plt.plot(historia.history['accuracy'])
plt.title('Precisión / iteraciones')
plt.ylabel('Precisión')
plt.xlabel('Iteraciones')

plt.show()

precision = modelo.evaluate(X_test, Y_test, verbose=0)
print('Precisión en el set de validación: {:.1f}%'.format(100*precision[1]))

Y_pred = modelo.predict_classes(X_test)

ids_imgs = np.random.randint(0,X_test.shape[0],9)
for i in range(len(ids_imgs)):
	idx = ids_imgs[i]
	img = X_test[idx,:].reshape(28,28)
	cat_original = np.argmax(Y_test[idx,:])
	cat_prediccion = Y_pred[idx]

	plt.subplot(3,3,i+1)
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	plt.title('"{}" = "{}"'.format(cat_original, cat_prediccion))
plt.suptitle('Ejemplos de clasificación en el set de validación')
plt.show()

