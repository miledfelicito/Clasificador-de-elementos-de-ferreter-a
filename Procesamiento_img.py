from matplotlib import pyplot as plt
import numpy as np
import cv2
import rembg
import os

def procesamiento(image):

    # Remover el fondo de la imagen usando rembg
    bg_removed_image = rembg.remove(image)
    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(bg_removed_image, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque gaussiano para reducir el ruido
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Aplicar umbralización de Otsu para binarizar la imagen 0 = negro, 255 = Blanco 
    _, thresholded_image_1 = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Aplicar umbralización adaptativa para mejorar la binarización
    adaptive_threshold = cv2.adaptiveThreshold(thresholded_image_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    return adaptive_threshold


# def procesar_carpeta(carpeta_origen, carpeta_destino):
#     # Verificar si la carpeta de destino existe, si no, crearla
#     if not os.path.exists(carpeta_destino):
#         os.makedirs(carpeta_destino)
    
#     # Recorrer todas las imágenes en la carpeta de origen
#     for filename in os.listdir(carpeta_origen):
#         if filename.endswith(".jpg") or filename.endswith(".png"):  # puedes agregar otros formatos si es necesario
#             image_path = os.path.join(carpeta_origen, filename)
#             image = cv2.imread(image_path)
#             if image is not None:
#                 processed_image = procesamiento(image)
#                 processed_image_path = os.path.join(carpeta_destino, filename)
#                 cv2.imwrite(processed_image_path, processed_image)

# # Especificar las rutas de las carpetas
# carpeta_origen = "D:\DB\IA_Proyecto\Data Base\YTrain\Tuercas_acotado"
# carpeta_destino =  "D:\DB\IA_Proyecto\Data Base\YTrain\Tuercas_acotado_proc"

# # Llamar a la función para procesar todas las imágenes en la carpeta
# procesar_carpeta(carpeta_origen, carpeta_destino)