import cv2
import mediapipe as mp

# Iniciar Mediapipe para detección de manos
mp_mano = mp.solutions.hands
mano = mp_mano.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9)
mp_dibujo = mp.solutions.drawing_utils

# Abrir la cámara
captura = cv2.VideoCapture(0)

# Si no se pudo abrir la cámara
if not captura.isOpened():
    #print("No se pudo abrir la cámara")
    exit()

# Mientras la cámara esté activa
while True:
    # Captura el frame
    ret, frame = captura.read()
    
    # Si no se obtuvo el frame, se rompe el bucle
    if not ret:
        #print("No se pudo recibir el frame (finalización del flujo)")
        break
    
    # Modo espejo
    frame = cv2.flip(frame, 1)
    
    # Dimensiones de imagen
    alto, ancho, _ = frame.shape
    
    # Área a leer
    inicio_x = int(0.6 * ancho)
    inicio_y = int(0.25 * alto)
    fin_x = int(0.9 * ancho)
    fin_y = int(0.75 * alto)
    
    # Dibujar rectángulo
    color = ((255, 0, 0))
    grosor = 2
    cv2.rectangle(frame, (inicio_x, inicio_y), (fin_x, fin_y), color, grosor)
    
    # Recortar la sección a analizar
    recorte = frame[inicio_y:fin_y, inicio_x:fin_x]
    
    # Convertir el recorte de BGR (por defecto de OpenCV) a RGB (para Mediapipe)
    recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen con Mediapipe
    resultado = mano.process(recorte_rgb)

    # Si se detecta una mano, dibuja los landmarks (líneas y puntos)
    if resultado.multi_hand_landmarks:
        for mano_landmarks in resultado.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(
                recorte,  # Dibujar sobre el recorte
                mano_landmarks,
                mp_mano.HAND_CONNECTIONS,
                mp_dibujo.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), # Color de puntos
                mp_dibujo.DrawingSpec(color=(0, 255, 255), thickness=2) # Color de líneas
            )
    
    # Vuelve a insertar el recorte procesado en el frame completo
    frame[inicio_y:fin_y, inicio_x:fin_x] = recorte

    # Mostrar frame
    cv2.imshow('Camara', frame)
    
    # Cerrar al presionar "Esc" (código ASCII = 27)
    if cv2.waitKey(1) == 27:
        break

# Finalizar
captura.release()
cv2.destroyAllWindows()