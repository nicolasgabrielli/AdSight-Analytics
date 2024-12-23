import cv2
from deepface import DeepFace
import tensorflow as tf
from threading import Thread
from queue import Queue
import time
import uuid
import math

# Configuración para utilizar la GPU si está disponible
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU detectada y configurada para su uso.")
        except RuntimeError as e:
            print(f"Error al configurar la GPU: {e}")
    else:
        print("No se detectaron GPUs, utilizando CPU.")

# Llamar a la configuración de la GPU
setup_gpu()

def distance(coord1, coord2):
    """
    Calcula la distancia euclidiana entre dos puntos.
    """
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def assign_persistent_id(persons, region, max_distance=100):
    """
    Asigna un ID persistente basado en la proximidad de las coordenadas detectadas.
    """
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    center_new = (x + w // 2, y + h // 2)

    for person_id, data in persons.items():
        center_existing = data["center"]
        if distance(center_existing, center_new) < max_distance:
            # Actualiza las coordenadas de la persona detectada
            persons[person_id]["center"] = center_new
            persons[person_id]["last_seen"] = time.time()
            return person_id

    # Si no se encuentra un ID cercano, crea uno nuevo
    new_id = str(uuid.uuid4())
    persons[new_id] = {
        "center": center_new,
        "time_in_screen": 0,
        "last_seen": time.time(),
        "age": "N/A",
        "gender": "N/A"
    }
    return new_id

def save_persons_to_file(persons, filename="persons_detected.txt"):
    """
    Guarda la información de las personas detectadas en un archivo .txt.
    """
    with open(filename, "w") as file:
        for person_id, data in persons.items():
            file.write(f"ID: {person_id[:8]}, Edad: {data.get('age', 'N/A')}, Género: {data.get('gender', 'N/A')}, Tiempo en pantalla: {data['time_in_screen']} segundos\n")
    print(f"Información guardada en {filename}")

def analyze_faces_in_realtime():
    # Inicializa la cámara
    vid = cv2.VideoCapture("rtsp://admin:Admin123.@192.168.1.100:554/profile2/media.smp")
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Diccionario para almacenar datos de las personas detectadas
    persons = {}

    # Cola para almacenar el último frame capturado
    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)

    def capture_frames():
        while True:
            ret, frame = vid.read()
            if not ret:
                print("No se pudo capturar el frame.")
                break
            if not frame_queue.full():
                frame_queue.put(frame)

    def process_frames():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                resized_frame = cv2.resize(frame, (640, 360))
                try:
                    results = DeepFace.analyze(
                        resized_frame,
                        actions=['age', 'gender'],
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    result_queue.put((frame, results))
                except Exception as e:
                    print(f"Error al analizar el rostro: {e}")
                    result_queue.put((frame, None))

    capture_thread = Thread(target=capture_frames, daemon=True)
    process_thread = Thread(target=process_frames, daemon=True)
    capture_thread.start()
    process_thread.start()

    while True:
        if not result_queue.empty():
            frame, results = result_queue.get()
            if results:
                if isinstance(results, list):
                    analyses = results
                else:
                    analyses = [results]

                for analysis in analyses:
                    region = analysis.get("region", {})
                    person_id = assign_persistent_id(persons, region)

                    # Actualizar datos de la persona detectada
                    persons[person_id]["age"] = analysis.get("age", "N/A")
                    persons[person_id]["gender"] = analysis.get("dominant_gender", "N/A")
                    persons[person_id]["time_in_screen"] += 1

                    # Dibujar bounding box y atributos
                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                    if w > 0 and h > 0:
                        cv2.rectangle(frame, (x * 2, y * 2), (x * 2 + w * 2, y * 2 + h * 2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {person_id[:8]}", (x * 2, y * 2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Edad: {analysis.get('age', 'N/A')}", (x * 2, y * 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Genero: {analysis.get('dominant_gender', 'N/A')}", (x * 2, y * 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Mostrar el frame con resultados
            cv2.imshow('Análisis Facial en Tiempo Real', frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

    # Guardar datos de las personas detectadas
    save_persons_to_file(persons)

# Ejecutar la función
if __name__ == "__main__":
    analyze_faces_in_realtime()
