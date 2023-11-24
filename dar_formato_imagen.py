import os
import cv2

directorio_imagenes = './PinesFOPT-OPT/validacion/FOPT'

nueva_resolucion = (2048, 1536)

for nombre_archivo in os.listdir(directorio_imagenes):
    if nombre_archivo.endswith(".jpg"):
        ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta_imagen)
        imagen_rotada = cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE)
        alto_rotado, ancho_rotado, _ = imagen_rotada.shape
        nuevo_alto = int((nueva_resolucion[0] / ancho_rotado) * alto_rotado)
        # Redimensionar la imagen
        imagen_redimensionada = cv2.resize(imagen_rotada, (nueva_resolucion[0], nuevo_alto))
        cv2.imwrite(ruta_imagen, imagen_redimensionada)
print("Proceso completado.")
