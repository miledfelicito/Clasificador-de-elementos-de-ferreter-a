from matplotlib import pyplot as plt
import numpy as np
import cv2

def calcular_longitud(image):

    # Llega la imagen ya procesada 
    processed_image = image

    # Mostrar la imagen procesada
    plt.imshow(processed_image, cmap='gray')
    plt.title("Imagen Procesada")
    plt.show()

    # Aplicar el detector de bordes Canny
    edges = cv2.Canny(processed_image, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si no se encuentran contornos, retornar 0
    if not contours:
        return 0

    # Encontrar el contorno con el área más grande (asumiendo que es el clavo o tornillo)
    longest_contour = max(contours, key=cv2.contourArea)

    # Calcular el área del contorno
    area = cv2.contourArea(longest_contour)
    print(f"Área del contorno: {area}")

    # Calcular el rectángulo delimitador del contorno
    x, y, w, h = cv2.boundingRect(longest_contour)
    print(f"Rectángulo delimitador: x = {x}, y = {y}, w = {w}, h = {h}")

    # Dibujar el contorno y el rectángulo delimitador en la imagen de entrada
    output_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, [longest_contour], -1, (0, 255, 0), 2)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar la imagen con contornos y rectángulo delimitador
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Contorno y Rectángulo Delimitador")
    plt.show()

    # La longitud del clavo o tornillo se asume como la dimensión más grande del rectángulo delimitador
    length = max(w, h)
    # Longitud en pixeles
    return length

