import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Función para preprocesar la imagen
def preprocess_and_extract_features(imagepath):
    try:
        image = cv2.imread(imagepath)
        if image is None:
            print(f'Error loading image: {imagepath}')
            return None

        # Convertir la imagen al espacio de color HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extraer histogramas para los canales H y S
        h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])

        # Extracción de características: primer momento de la distribución
        h_mean = np.mean(h_hist)
        s_mean = np.mean(s_hist)

        return h_mean, s_mean, imagepath  # Devolver la ruta de la imagen junto con las características

    except Exception as e:
        print(f'Error al procesar la imagen {imagepath}: {e}')
        return None


def mbsas(featuresm, threshold, min_pts):
    clusters = []
    for featurem in featuresm:
        added_to_cluster = False
        for clusterm in clusters:
            for cluster_feature in clusterm:
                try:
                    diff = np.array(featurem[:-1], dtype=np.float32) - np.array(cluster_feature[:-1], dtype=np.float32)
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff < threshold:
                        clusterm.append(featurem)
                        added_to_cluster = True
                        break
                except Exception as e:
                    print(f"Error al comparar caracteristicas: {e}")
                    print(f"featurem[:-1]: {featurem[:-1]}, cluster_feature[:-1]: {cluster_feature[:-1]}")

        if not added_to_cluster:
            new_cluster = [featurem]
            clusters.append(new_cluster)

    # Filtrar clusters con puntos por debajo del umbral mínimo
    clusters = [clusterm for clusterm in clusters if len(clusterm) >= min_pts]

    return clusters


image_directory = './PinesFOPT-OPT/entrenamiento/FOPT'

features_list = []

# Iterar sobre las imágenes en el directorio
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_directory, filename)
        features = preprocess_and_extract_features(image_path)

        if features is not None:
            features_list.append(features)

# Convertir la lista de características a un array de numpy 20 y 10
features = np.array(features_list)
threshold_value = 20
min_points = 10
cluster_labels = []
result_clusters = mbsas(features, threshold_value, min_points)

ground_truth = {'./PinesFOPT-OPT/validacion/FOPT/28_cuatro.jpg': 'Madura',
                './PinesFOPT-OPT/validacion/FOPT/2_uno.jpg': 'Verde', './PinesFOPT-OPT/validacion/FOPT/24_dos.jpg':
                    'Medio-Madura'}

# Asignar etiquetas a cada característica dentro del cluster basándose en un esquema de votación mayoritaria
for cluster in result_clusters:
    for feature in cluster:
        _, _, image_path = feature  # Desempaquetar los valores de la característica
        true_label = ground_truth.get(image_path, 'unknown')
        label_counts = {label: 0 for label in set(ground_truth.values())}

        for other_feature in cluster:
            _, _, other_image_path = other_feature
            other_true_label = ground_truth.get(other_image_path, 'unknown')
            if other_true_label in label_counts:
                label_counts[other_true_label] += 1

        # Asignar la etiqueta con el recuento más alto a la característica individual
        dominant_label = max(label_counts, key=label_counts.get)
        cluster_labels.append(dominant_label)

# Mostrar resultados
for i, cluster in enumerate(result_clusters):
    print(f"Cluster {i + 1}:")
    print(f"  Numero de imagenes en el cluster: {len(cluster)}")

    # Visualizar algunas imágenes del cluster
    for j, feature in enumerate(cluster[:min(5, len(cluster))]):
        image_path = feature[-1]
        assigned_label = cluster_labels[i * len(cluster) + j]

        # Mostrar la imagen junto con las etiquetas
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.title(f"Cluster {i + 1}, Imagen {j + 1}\nEstado: {assigned_label}")
        plt.axis('off')
        plt.show()
    print("\n")
