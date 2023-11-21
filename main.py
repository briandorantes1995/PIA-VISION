import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt


# Funciones de la red neuronal
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Error al cargar la imagen: {image_path}')
            return None
        resized_image = cv2.resize(image, (width, height))
        flattened_image = resized_image.flatten()
        normalized_image = flattened_image / 255.0
        return normalized_image
    except Exception as e:
        print(f'Error al procesar la imagen {image_path}: {e}')
        return None



# Funciones de la red neuronal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize_weights(inputsize, hiddensize, outputsize):
    np.random.seed(42)
    weightsinput_hidden = 2 * np.random.random((inputsize, hiddensize)) - 1
    weightshidden_output = 2 * np.random.random((hiddensize, outputsize)) - 1
    return weightsinput_hidden, weightshidden_output


def train_neural_network(xtrain, ytrain, hiddensize, epochtrain, learningrate):
    input_size = xtrain.shape[1]
    output_size = 1

    wih, who = initialize_weights(input_size, hiddensize, output_size)

    for epoch in range(epochtrain):
        # Capa oculta
        hidden_layer_input = np.dot(xtrain, wih)
        hidden_layer_output = sigmoid(hidden_layer_input)

        # Capa de salida
        output_layer_input = np.dot(hidden_layer_output, who)
        predicted_output = sigmoid(output_layer_input)

        # Cálculo del error
        error = ytrain - predicted_output

        # Retropropagación
        output_error = error * sigmoid_derivative(predicted_output)
        hidden_layer_error = output_error.dot(who.T) * sigmoid_derivative(hidden_layer_output)

        # Actualización de pesos
        who += hidden_layer_output.T.dot(output_error) * learningrate
        wih += X_train.T.dot(hidden_layer_error) * learningrate

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')

    return wih, who


def predict(x, weights_inputhidden, weights_hiddenoutput):
    hidden_layer_input = np.dot(x, weights_inputhidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hiddenoutput)
    predicted_output = sigmoid(output_layer_input)

    return predicted_output


# Parámetros
hidden_size = 4
epochs = 10000
learning_rate = 0.1
width, height = 64, 64

# Directorios de las imágenes
directory_FOPT = './PinesFOPT-OPT/entrenamiento/FOPT/'
directory_OPT = './PinesFOPT-OPT/entrenamiento/OPT/'
directory_FOPT_val = './PinesFOPT-OPT/validacion/FOPT/'
directory_OPT_val = './PinesFOPT-OPT/validacion/OPT/'

# Lista de imágenes FOPT
lista_de_imagenes_FOPT = [f for f in os.listdir(directory_FOPT) if os.path.isfile(os.path.join(directory_FOPT, f))]

# Lista de imágenes OPT
lista_de_imagenes_OPT = [f for f in os.listdir(directory_OPT) if os.path.isfile(os.path.join(directory_OPT, f))]

# Lista de imágenes FOPT para validación
lista_de_imagenes_FOPT_val = [f for f in os.listdir(directory_FOPT_val) if os.path.isfile(os.path.join
                                                                                          (directory_FOPT_val, f))]

# Lista de imágenes OPT para validación
lista_de_imagenes_OPT_val = [f for f in os.listdir(directory_OPT_val) if os.path.isfile(os.path.join(directory_OPT_val, f))]


# Para datos de entrenamiento
X_train = np.array([preprocess_image('./PinesFOPT-OPT/entrenamiento/FOPT/'+img) for img in lista_de_imagenes_FOPT])
y_train = np.ones(len(lista_de_imagenes_FOPT))  # Etiqueta 1 para FOPT (madura)

X_train = np.concatenate([X_train, np.array(
    [preprocess_image('./PinesFOPT-OPT/entrenamiento/OPT/'+img) for img in lista_de_imagenes_OPT])])
y_train = np.concatenate([y_train, np.zeros(len(lista_de_imagenes_OPT))])  # Etiqueta 0 para OPT (inmadura)

weights_input_hidden, weights_hidden_output = train_neural_network(X_train, y_train.reshape(-1, 1), hidden_size, epochs,
                                                                   learning_rate)

X_val_FOPT = np.array(
    [preprocess_image('./PinesFOPT-OPT/validacion/FOPT/'+img) for img in lista_de_imagenes_FOPT_val])
X_val_OPT = np.array([preprocess_image('./PinesFOPT-OPT/validacion/OPT/'+img) for img in lista_de_imagenes_OPT_val])

predictions_FOPT = predict(X_val_FOPT, weights_input_hidden, weights_hidden_output)
predictions_OPT = predict(X_val_OPT, weights_input_hidden, weights_hidden_output)

# Umbral de decisión para clasificar como madura o no madura
threshold = 0.5
classifications_FOPT = predictions_FOPT > threshold
classifications_OPT = predictions_OPT > threshold

# Pruebas
# Elegir una imagen al azar de FOPT o OPT para la demostración
random_image_FOPT = random.choice(lista_de_imagenes_FOPT_val)
random_image_OPT = random.choice(lista_de_imagenes_OPT_val)

# Ruta completa de la imagen elegida
image_path_FOPT = os.path.join(directory_FOPT_val, random_image_FOPT)
image_path_OPT = os.path.join(directory_OPT_val, random_image_OPT)

# Leer la imagen
image_FOPT = cv2.imread(image_path_FOPT)
image_OPT = cv2.imread(image_path_OPT)

# Preprocesar la imagen para la predicción
processed_image_FOPT = preprocess_image(image_path_FOPT)
processed_image_OPT = preprocess_image(image_path_OPT)

# Realizar la predicción
prediction_FOPT = predict(processed_image_FOPT, weights_input_hidden, weights_hidden_output)
prediction_OPT = predict(processed_image_OPT, weights_input_hidden, weights_hidden_output)

# Clasificar según la predicción
classification_FOPT = "Madura" if prediction_FOPT > threshold else "Rechazada"
classification_OPT = "Madura" if prediction_OPT > threshold else "Rechazada"

# Mostrar la imagen con leyenda
plt.figure(figsize=(10, 5))

# Imagen FOPT
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_FOPT, cv2.COLOR_BGR2RGB))
plt.title(f"FOPT - Clasificación: {classification_FOPT}")
plt.axis('off')

# Imagen OPT
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_OPT, cv2.COLOR_BGR2RGB))
plt.title(f"OPT - Clasificación: {classification_OPT}")
plt.axis('off')

plt.show()


# Display the original image and the segmented images
original_image = cv2.imread(random_image_path)
cv2.imshow('Original Image', original_image)
cv2.imshow('H Threshold', h_threshold)
cv2.imshow('S Threshold', s_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
