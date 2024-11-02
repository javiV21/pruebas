import cv2
import numpy as np
import os
import mediapipe as mp

# Configuraciones de Mediapipe
mp_manos = mp.solutions.hands
manos = mp_manos.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9)
mp_dibujo = mp.solutions.drawing_utils

# Abrir la cámara
captura = cv2.VideoCapture(0)

# Verifica cámara abierta
if not captura.isOpened():
    exit()

# Cargar imágenes del dataset
dataset_path = 'img3'  # Cambia esto por la ruta a tu dataset
dataset_imagenes = []
nombres_imagenes = []

# Cargar todas las imágenes en el dataset
for nombre_imagen in os.listdir(dataset_path):
    if nombre_imagen.endswith('.jpeg') or nombre_imagen.endswith('.jpg'):
        imagen = cv2.imread(os.path.join(dataset_path, nombre_imagen))
        dataset_imagenes.append(imagen)
        nombres_imagenes.append(nombre_imagen)

while True:
    ret, frame = captura.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    alto, ancho, _ = frame.shape
    
    # Definir área de interés
    inicio_x = int(0.2 * ancho)
    inicio_y = int(0.25 * alto)
    fin_x = int(0.5 * ancho)
    fin_y = int(0.75 * alto)

    # Dibujar rectángulo
    cv2.rectangle(frame, (inicio_x, inicio_y), (fin_x, fin_y), (255, 0, 0), 2)
    
    recorte = frame[inicio_y:fin_y, inicio_x:fin_x]
    recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
    resultados = manos.process(recorte_rgb)

    if resultados.multi_hand_landmarks:
        for mano_landmarks in resultados.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(
                recorte, 
                mano_landmarks,
                mp_manos.HAND_CONNECTIONS,
                mp_dibujo.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  
                mp_dibujo.DrawingSpec(color=(0, 255, 255), thickness=2)  
            )
    
        # Almacenar imagen recortada para comparación
        recorte_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        recorte_gray = cv2.resize(recorte_gray, (100, 100))  # Redimensionar para la comparación
        
        mejor_similitud = 0
        mejor_nombre = None

        # Comparar con las imágenes del dataset
        for i, imagen in enumerate(dataset_imagenes):
            imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen_gray = cv2.resize(imagen_gray, (100, 100))  # Redimensionar

            # Calcular la similitud de histograma
            hist_recorte = cv2.calcHist([recorte_gray], [0], None, [256], [0, 256])
            hist_imagen = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256])
            cv2.normalize(hist_recorte, hist_recorte)
            cv2.normalize(hist_imagen, hist_imagen)
            similitud = cv2.compareHist(hist_recorte, hist_imagen, cv2.HISTCMP_CORREL)  # Similaridad

            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor_nombre = nombres_imagenes[i]

        if mejor_nombre is not None:
            cv2.putText(frame, f"Mejor coincidencia: {mejor_nombre}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    frame[inicio_y:fin_y, inicio_x:fin_x] = recorte
    cv2.imshow('Captura de imagen', frame)

    if cv2.waitKey(1) == 27:
        break

captura.release()
cv2.destroyAllWindows()