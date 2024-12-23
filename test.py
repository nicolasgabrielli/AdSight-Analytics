import cv2
import mediapipe as mp
import time
import threading
import math
import uuid
import numpy as np
from deepface import DeepFace

# Inicializar herramientas de MediaPipe
mp_deteccion_rostros = mp.solutions.face_detection
mp_dibujo = mp.solutions.drawing_utils

# Configurar la detección facial con MediaPipe
deteccion_rostros = mp_deteccion_rostros.FaceDetection(min_detection_confidence=0.9)

# Inicializar la cámara
url_rtsp = "rtsp://admin:Admin123.@192.168.1.100:554/profile2/media.smp"  # URL RTSP de la cámara

# En caso de usar la url:
#captura_video = cv2.VideoCapture(url_rtsp)

# Para utilizar la cámara del PC
captura_video = cv2.VideoCapture(0)

# Variables para medir el tiempo
personas_tiempo = {}
id_detalles = {}
ids_activas = {}

# Variables para procesamiento en paralelo
frame = None
procesando = False
frame_counter = 0

# Cargar modelos preentrenados de OpenCV para género y edad
modelo_edad = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
modelo_genero = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

# Lista de rangos de edad y géneros
RANGOS_EDAD = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-']
GENEROS = ['Femenino', 'Masculino']

# Función para capturar frames continuamente
def capturar_frames():
    global frame
    while True:
        ret, nuevo_frame = captura_video.read()
        if ret:
            frame = nuevo_frame

# Iniciar hilo para capturar frames
hilo_captura = threading.Thread(target=capturar_frames, daemon=True)
hilo_captura.start()

# Limitar el procesamiento a un número específico de FPS
fps_limit = 10
prev_time = 0

# Función para asignar ID único a un rostro basado en la ubicación de la caja delimitadora
def asignar_id_unico(bounding_box, ids_activas):
    distancia_minima = 50  # Ajustar umbral de proximidad (en píxeles)
    x, y, ancho, alto = bounding_box
    centro_x_nuevo = x + ancho // 2
    centro_y_nuevo = y + alto // 2

    for id_activo, datos in ids_activas.items():
        (x_existente, y_existente, ancho_existente, alto_existente), ultima_vista = datos
        centro_x_existente = x_existente + ancho_existente // 2
        centro_y_existente = y_existente + alto_existente // 2

        distancia = math.sqrt((centro_x_nuevo - centro_x_existente) ** 2 + (centro_y_nuevo - centro_y_existente) ** 2)
        if distancia < distancia_minima:
            ids_activas[id_activo] = (bounding_box, time.time())
            return id_activo

    nuevo_id = str(uuid.uuid4())
    ids_activas[nuevo_id] = (bounding_box, time.time())
    return nuevo_id

# Limpiar IDs inactivos
def limpiar_ids_inactivos(ids_activas, tiempo_inactivo=2):
    tiempo_actual = time.time()
    ids_a_eliminar = [id_activo for id_activo, (_, ultima_vista) in ids_activas.items() if tiempo_actual - ultima_vista > tiempo_inactivo]
    for id_activo in ids_a_eliminar:
        del ids_activas[id_activo]

# Analizar género y edad utilizando los modelos de OpenCV
def analizar_genero_y_edad_OG(frame, bounding_box):
    x, y, ancho, alto = bounding_box
    if x < 0 or y < 0 or x + ancho > frame.shape[1] or y + alto > frame.shape[0]:
        return "Desconocido", "N/A"

    # Ajustar el ROI con margen
    margen = 10
    x = max(x - margen, 0)
    y = max(y - margen, 0)
    ancho = min(ancho + 2 * margen, frame.shape[1] - x)
    alto = min(alto + 2 * margen, frame.shape[0] - y)
    rostro = frame[y:y + alto, x:x + ancho]

    # Asegurar tamaño adecuado
    if rostro.shape[0] > 0 and rostro.shape[1] > 0:
        blob = cv2.dnn.blobFromImage(rostro, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predicción de género
        modelo_genero.setInput(blob)
        pred_genero = modelo_genero.forward()
        probabilidad_genero = pred_genero[0].max()
        genero = GENEROS[pred_genero[0].argmax()] if probabilidad_genero > 0.6 else "Desconocido"

        # Predicción de edad
        modelo_edad.setInput(blob)
        pred_edad = modelo_edad.forward()
        edad_predicha = RANGOS_EDAD[pred_edad[0].argmax()]



    # Asegurar que el rostro tenga el tamaño y formato esperado
    if rostro.shape[0] == 0 or rostro.shape[1] == 0:
        return "Desconocido", "N/A"

    try:
        blob = cv2.dnn.blobFromImage(rostro, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predicción de género
        modelo_genero.setInput(blob)
        pred_genero = modelo_genero.forward()
        genero = GENEROS[pred_genero[0].argmax()]

        # Predicción de edad
        modelo_edad.setInput(blob)
        pred_edad = modelo_edad.forward()
        edad = RANGOS_EDAD[pred_edad[0].argmax()]

        return genero, edad
    except Exception as e:
        print(f"Error en análisis de género y edad: {e}")
        return "Desconocido", "N/A"

def analizar_genero_y_edad(frame, bounding_box):
    x, y, ancho, alto = bounding_box

    # Validar límites del ROI (bounding box)
    if x < 0 or y < 0 or x + ancho > frame.shape[1] or y + alto > frame.shape[0]:
        print("ROI fuera de los límites del frame.")
        return "Desconocido", "N/A"

    # Recortar el rostro
    rostro = frame[y:y + alto, x:x + ancho]

    # Validar dimensiones del ROI
    if rostro is None or rostro.shape[0] < 10 or rostro.shape[1] < 10:
        print("ROI inválido o muy pequeño.")
        return "Desconocido", "N/A"

    # Convertir al formato RGB (DeepFace requiere imágenes en RGB)
    rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)

    try:
        # Analizar género y edad
        analisis = DeepFace.analyze(rostro_rgb, actions=['gender', 'age'], enforce_detection=True)
        genero = analisis['gender']
        edad = analisis['age']
        return genero, str(int(edad))
    except Exception as e:
        print(f"Error en DeepFace: {e}")
        return "Desconocido", "N/A"

try:
    while True:
        # Controlar la tasa de procesamiento
        current_time = time.time()
        if current_time - prev_time < 1 / fps_limit:
            continue
        prev_time = current_time

        if frame is None:
            continue

        frame_counter += 1

        # Convertir el frame a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar rostros
        resultados = deteccion_rostros.process(frame_rgb)

        if resultados.detections:
            for deteccion in resultados.detections:
                box = deteccion.location_data.relative_bounding_box
                h, w, _ = frame.shape
                # Ajusta el margen en píxeles (puedes probar diferentes valores)
                margen = 20

                # Calcula las nuevas coordenadas ampliadas
                x = max(int(box.xmin * w) - margen, 0)
                y = max(int(box.ymin * h) - margen, 0)
                ancho = min(int(box.width * w) + margen * 2, w - x)
                alto = min(int(box.height * h) + margen * 2, h - y)

                bounding_box = (x, y, ancho, alto)


                id_rostro = asignar_id_unico(bounding_box, ids_activas)

                if id_rostro not in personas_tiempo:
                    personas_tiempo[id_rostro] = 0
                    genero, edad = analizar_genero_y_edad(frame, bounding_box)
                    id_detalles[id_rostro] = {'género': genero, 'edad': edad}

                # Incrementar el tiempo de visualización para el rostro identificado
                personas_tiempo[id_rostro] += 1 / fps_limit

                # Dibujar el rostro, ID único, género y edad
                cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
                detalles = f"ID: {id_rostro[:8]} | {id_detalles[id_rostro]['género']} | {id_detalles[id_rostro]['edad']}"
                cv2.putText(frame, detalles, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Limpiar IDs inactivos
        limpiar_ids_inactivos(ids_activas)

        # Mostrar el video en una ventana
        cv2.imshow('Detección y Reconocimiento de Rostros', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Guardar tiempos y detalles en un archivo
    with open("tiempos_visualizacion.txt", "w") as archivo:
        for id_rostro, tiempo in personas_tiempo.items():
            if 1 <= tiempo <= 600:  # Descartar tiempos menores a 1 segundo o mayores a 10 minutos
                detalles = id_detalles.get(id_rostro, {'género': 'Desconocido', 'edad': 'N/A'})
                archivo.write(f"ID: {id_rostro[:8]} | Género: {detalles['género']} | Edad: {detalles['edad']} | Tiempo: {tiempo:.4f} segundos\n")

    # Liberar la cámara y cerrar ventanas
    captura_video.release()
    cv2.destroyAllWindows()
