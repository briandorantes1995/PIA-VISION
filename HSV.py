import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Function to preprocess image
def preprocess_and_extract_features(imagepath):
    try:
        image = cv2.imread(imagepath)
        if image is None:
            print(f'Error loading image: {imagepath}')
            return None

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract histograms for H and S channels
        h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])

        # Feature extraction: first moment of distribution
        h_mean = np.mean(h_hist)
        s_mean = np.mean(s_hist)

        return h_mean, s_mean, imagepath  # Return image path along with features

    except Exception as e:
        print(f'Error processing image {imagepath}: {e}')
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
                    print(f"Error comparing features: {e}")
                    print(f"featurem[:-1]: {featurem[:-1]}, cluster_feature[:-1]: {cluster_feature[:-1]}")

        if not added_to_cluster:
            new_cluster = [featurem]
            clusters.append(new_cluster)

    # Filter clusters with points below the minimum threshold
    clusters = [clusterm for clusterm in clusters if len(clusterm) >= min_pts]

    return clusters


# Example usage
# Directory containing your images
image_directory = './PinesFOPT-OPT/entrenamiento/FOPT/'

# List to store features for each image
features_list = []

# Iterate over images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_directory, filename)
        features = preprocess_and_extract_features(image_path)

        if features is not None:
            features_list.append(features)

# Convert the list of features to a NumPy array
features = np.array(features_list)
threshold_value = 20
min_points = 10

result_clusters = mbsas(features, threshold_value, min_points)

# Suppose you have a ground truth for a subset of images
ground_truth = {'./PinesFOPT-OPT/validacion/FOPT/38_cuatro.jpg': 'Madura',
                './PinesFOPT-OPT/validacion/FOPT/2_uno.jpg': 'Verde', './PinesFOPT-OPT/validacion/FOPT/25_dos.jpg':
                    'Medio-Madura'}

# Assign labels to clusters based on a majority voting scheme
cluster_labels = []
for cluster in result_clusters:
    label_counts = {label: 0 for label in set(ground_truth.values())}
    for feature in cluster:
        _, _, image_path = feature  # Desempaquetar los valores de la característica
        true_label = ground_truth.get(image_path, 'unknown')  # Usar get para proporcionar un valor predeterminado
        if true_label in label_counts:  # Verificar si la etiqueta está en el conjunto de etiquetas
            label_counts[true_label] += 1

    # Assign the label with the highest count to the cluster
    dominant_label = max(label_counts, key=label_counts.get)
    cluster_labels.append(dominant_label)


# Mostrar resultados
for i, cluster in enumerate(result_clusters):
    print(f"Cluster {i + 1}:")
    print(f"  Número de imágenes en el cluster: {len(cluster)}")

    # Visualizar algunas imágenes del cluster
    for j, feature in enumerate(cluster[:min(3, len(cluster))]):  # Visualiza hasta 3 imágenes de cada cluster
        image_path = feature[-1]  # Last element is the image_path
        assigned_label = cluster_labels[i]  # Use the cluster label assigned during majority voting

        # Obtener la etiqueta real desde el diccionario de ground_truth
        true_label = ground_truth.get(image_path, 'Desconocido')

        # Mostrar la imagen junto con las etiquetas
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.title(f"Cluster {i + 1}, Imagen {j + 1}\nEtiqueta asignada: {assigned_label}, Etiqueta real: {true_label}")
        plt.axis('off')  # Desactivar ejes
        plt.show()
    print("\n")
