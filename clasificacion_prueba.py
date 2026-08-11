import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['image.cmap'] = 'gray'
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, color, img_as_float, filters
from skimage.feature import hog
import cv2
from Procesamiento_img import procesamiento
from Medicion_prueba import calcular_longitud

def calcular_area_perimetro(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Calculo area de contorno mas grande y con eso el perimetro
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        perimeter = cv2.arcLength(max_contour, True)
        if perimeter != 0:
            # 4*pi*area / perimetro^2
            circularidad = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularidad = 0
    else:
        area = 0
        perimeter = 0
        circularidad = 0
    return area, perimeter, circularidad

def extraccion(image):
    hu = cv2.HuMoments(cv2.moments(image)).flatten()
    return image, [hu[0], hu[1], hu[3]]

# Analisis de la base de datos (YTrain)
# Entrenamiento de la base de datos 
# Hace una lista de las imagenes de la carpeta
tornillo = io.ImageCollection('D:\Facultad\IA1\IA_Proyecto\Data Base\YTrain\Tornillos_acotado_proc\*.jpg') 
tuerca = io.ImageCollection('D:\Facultad\IA1\IA_Proyecto\Data Base\YTrain\Tuercas_acotado_proc\*.jpg')
arandela = io.ImageCollection('D:\Facultad\IA1\IA_Proyecto\Data Base\YTrain\Arandelas_acotado_proc\*.jpg')
clavo = io.ImageCollection('D:\Facultad\IA1\IA_Proyecto\Data Base\YTrain\Clavos_acotado_proc\*.jpg')

# CLASE ELEMENTO
class Elemento:
    def __init__(self):
        self.pieza = None
        self.image = None
        self.caracteristica = []
        self.distancia = 0
        self.area = 0
        self.perimetro = 0
        self.circularidad = 0

# # Analisis de datos
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d') # 111 -> 1 fila, 1 columna, 1 grafico 3D

# i = 0

# Extraer caracteristicas de las imagenes 
def analizar_datos(coleccion, tipo, datos):
    for objeto in coleccion:
        elemento = Elemento()
        elemento.pieza = tipo
        elemento.image, elemento.caracteristica = extraccion(objeto)
        elemento.area, elemento.perimetro, elemento.circularidad = calcular_area_perimetro(elemento.image)
        datos.append(elemento)

    print("Todo joya")

# Plotear datos
def plot_datos(datos, ax):
    colores = {'Tornillo': 'yellow', 'Tuerca': 'red', 'Arandela': 'blue', 'Clavo': 'green'}
    for elemento in datos:
        ax.scatter(elemento.caracteristica[0], elemento.caracteristica[1], elemento.caracteristica[2],
                   c=colores[elemento.pieza], label=elemento.pieza)
    handles = [mpatches.Patch(color=color, label=label) for label, color in colores.items()]
    plt.legend(handles=handles)

# Analisis de datos
datos = []
analizar_datos(io.ImageCollection('D:\Facultad\IA1\IA_Proyecto\Data Base\YTrain\Tornillos_acotado_proc\*.jpg'), 'Tornillo', datos)
analizar_datos(io.ImageCollection('D:\Facultad\IA1\IA_Proyecto\Data Base\YTrain\Tuercas_acotado_proc\*.jpg'), 'Tuerca', datos)
analizar_datos(io.ImageCollection('D:\Facultad\IA1\IA_Proyecto\Data Base\YTrain\Arandelas_acotado_proc\*.jpg'), 'Arandela', datos)
analizar_datos(io.ImageCollection('D:\Facultad\IA1\IA_Proyecto\Data Base\YTrain\Clavos_acotado_proc\*.jpg'), 'Clavo', datos)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_datos(datos, ax)

ax.grid(True)
ax.set_title("Analisis completo de YTrain")

ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_zlabel('componente 4')

plt.show()

print("Analisis completo de la base de datos de YTrain")
print("Cantidad de imagenes analizadas: ")
print(len(datos))

# Elemento a evaluar
test = Elemento()
numero = input("Introduce numero de la foto: ")
nombre = './Data Base/Evaluacion2.0/Foto' + str(numero) + '.jpg'
# nombre = "D:\DB\IA_Proyecto\Data Base\Arandela3.jpg"
image = io.imread(nombre)
image_procesada = procesamiento(image)

# Extraigo caracteristicas de imagen a evaluar 
test.image, test.caracteristica = extraccion(image_procesada)
test.area, test.perimetro, test.circularidad = calcular_area_perimetro(test.image)
test.pieza = 'Arandela' # label inicial, despues cambia si es necesario

ax.scatter(test.caracteristica[0], test.caracteristica[1], test.caracteristica[2], c='k', marker='o') #
fig

# KNN
print("\nInicializacion KNN")
i = 0
sum = 0

# Verifica que la lista datos no este vacía
if len(datos) > 0:
    for ft in datos[0].caracteristica:
        sum = sum + np.power(np.abs(test.caracteristica[i] - ft), 2)
        i += 1
    d = np.sqrt(sum)
else:
    print("La lista 'datos' está vacía, no se pueden realizar operaciones.")

for element in datos:
    sum = 0
    i = 0
    for ft in (element.caracteristica):
        sum = sum + np.power(np.abs((test.caracteristica[i]) - ft), 2)
        i += 1

    # Incorporar el cálculo de distancia para área y perímetro
    sum += np.power(np.abs(test.area - element.area), 2)
    sum += np.power(np.abs(test.perimetro - element.perimetro), 2)
    sum += np.power(np.abs(test.circularidad - element.circularidad), 2)

    element.distancia = np.sqrt(sum)

    if (sum < d):
        d = sum
        test.pieza = element.pieza

print("Prediccion para KNN con K=1: ")
print(test.pieza)

# Algoritmo de ordenamiento de burbuja
# Si la distancia del elemento actual es mayor que la del siguiente, intercambiar los elementos
swap = True
while (swap):
    swap = False
    for i in range(1, len(datos) - 1):
        if (datos[i - 1].distancia > datos[i].distancia):
            aux = datos[i]
            datos[i] = datos[i - 1]
            datos[i - 1] = aux
            swap = True
print("\nPredicciones para KNN con K=5: ")
k = 5
for i in range(k):
    print(datos[i].pieza)

# K MEANS
import random
print("\nInicializacion KMeans")

tornillo_data = [d for d in datos if d.pieza == 'Tornillo']
tuerca_data = [d for d in datos if d.pieza == 'Tuerca']
arandela_data = [d for d in datos if d.pieza == 'Arandela']
clavo_data = [d for d in datos if d.pieza == 'Clavo']

def inicializar_media(data):
    elemento = random.choice(data)
    return list(elemento.caracteristica) + [elemento.area, elemento.perimetro, elemento.circularidad]

tornillo_mean = inicializar_media(tornillo_data)
tuerca_mean = inicializar_media(tuerca_data)
arandela_mean = inicializar_media(arandela_data)
clavo_mean = inicializar_media(clavo_data)

fig_means = plt.figure()
ax = fig_means.add_subplot(111, projection='3d')

ax.scatter(tornillo_mean[0], tornillo_mean[1], tornillo_mean[2], c='y', marker='o')
ax.scatter(tuerca_mean[0], tuerca_mean[1], tuerca_mean[2], c='r', marker='o')
ax.scatter(arandela_mean[0], arandela_mean[1], arandela_mean[2], c='b', marker='o')
ax.scatter(clavo_mean[0], clavo_mean[1], clavo_mean[2], c='g', marker='o')

ax.grid(True)
ax.set_title("Means")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label='Clavo')
plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_zlabel('componente 4')

plt.show()

def calcular_distancia(elemento, mean):
    distancia = 0
    for i in range(len(elemento.caracteristica)):
        distancia += np.power(np.abs(elemento.caracteristica[i] - mean[i]), 2)
    distancia += np.power(np.abs(elemento.area - mean[-3]), 2)
    distancia += np.power(np.abs(elemento.perimetro - mean[-2]), 2)
    distancia += np.power(np.abs(elemento.circularidad - mean[-1]), 2)
    return np.sqrt(distancia)

def actualizar_media(data):
    if len(data) > 0:
        return np.mean([d.caracteristica + [d.area, d.perimetro, d.circularidad] for d in data], axis=0).tolist()
    return []

# Asignacion, Actualizacion y Convergencia
tornillo_flag = True
tuerca_flag = True
arandela_flag = True
clavo_flag = True
iteracion = 0

while (tornillo_flag or tuerca_flag or arandela_flag or clavo_flag):
    tornillo_data = []
    tuerca_data = []
    arandela_data = []
    clavo_data = []

    for element in datos:
        distancia_tornillo = calcular_distancia(element, tornillo_mean)
        distancia_tuerca = calcular_distancia(element, tuerca_mean)
        distancia_arandela = calcular_distancia(element, arandela_mean)
        distancia_clavo = calcular_distancia(element, clavo_mean)
        
        dist_min = min(distancia_tornillo, distancia_tuerca, distancia_arandela, distancia_clavo)
        
        if dist_min == distancia_tornillo:
            tornillo_data.append(element)
        elif dist_min == distancia_tuerca:
            tuerca_data.append(element)
        elif dist_min == distancia_arandela:
            arandela_data.append(element)
        else:
            clavo_data.append(element)

    nueva_tornillo_mean = actualizar_media(tornillo_data)
    nueva_tuerca_mean = actualizar_media(tuerca_data)
    nueva_arandela_mean = actualizar_media(arandela_data)
    nueva_clavo_mean = actualizar_media(clavo_data)

    tornillo_flag = not np.array_equal(tornillo_mean, nueva_tornillo_mean)
    tuerca_flag = not np.array_equal(tuerca_mean, nueva_tuerca_mean)
    arandela_flag = not np.array_equal(arandela_mean, nueva_arandela_mean)
    clavo_flag = not np.array_equal(clavo_mean, nueva_clavo_mean)

    tornillo_mean = nueva_tornillo_mean
    tuerca_mean = nueva_tuerca_mean
    arandela_mean = nueva_arandela_mean
    clavo_mean = nueva_clavo_mean

    iteracion += 1

print(f"\nKMeans Convergió en {iteracion} iteraciones")

print(f"Tornillo media: {tornillo_mean}")
print(f"Tuerca media: {tuerca_mean}")
print(f"Arandela media: {arandela_mean}")
print(f"Clavo media: {clavo_mean}")

# Graficar resultados finales
fig_final = plt.figure()
ax_final = fig_final.add_subplot(111, projection='3d')

# Colores finales
ax_final.scatter(tornillo_mean[0], tornillo_mean[1], tornillo_mean[2], c='y', marker='o')
ax_final.scatter(tuerca_mean[0], tuerca_mean[1], tuerca_mean[2], c='r', marker='o')
ax_final.scatter(arandela_mean[0], arandela_mean[1], arandela_mean[2], c='b', marker='o')
ax_final.scatter(clavo_mean[0], clavo_mean[1], clavo_mean[2], c='g', marker='o')

ax_final.grid(True)
ax_final.set_title("KMeans Resultados Finales")

plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

print("\nPrediccion para KMeans: ")
print(test.pieza)

ax_final.set_xlabel('componente 1')
ax_final.set_ylabel('componente 2')
ax_final.set_zlabel('componente 4')

plt.show()

# Si la pieza es tornillo o clavo calcula la longitud
if test.pieza == "Tornillo" or test.pieza == "Clavo":
    # 20/618 es la relacion entre pixeles y mm  
    medida = calcular_longitud(image_procesada) * (20/618)
    print("La medida es " + str(medida) +" mm")
