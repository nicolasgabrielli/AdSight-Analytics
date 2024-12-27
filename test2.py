import cv2
from deepface import DeepFace
import tensorflow as tf
from threading import Thread, Lock
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
                tf.config.experimental.set_memory_growth(gpu, True)  # Usar método experimental
            print(f"GPU detectada y configurada: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"Error al configurar la GPU: {e}")
    else:
        print("No se detectaron GPUs, utilizando CPU como predeterminado.")

# Llamar a la configuración de la GPU
setup_gpu()

def distance(coord1, coord2):
    """
    Calcula la distancia euclidiana entre dos puntos.
    """
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def assign_persistent_id(persons, region, lock, max_distance=100):
    """
    Asigna un ID persistente basado en la proximidad de las coordenadas detectadas.
    """
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    center_new = (x + w // 2, y + h // 2)

    with lock:  # Sincroniza el acceso al diccionario
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
            "gender": "N/A",
            "race": "N/A",
            "emotion": "N/A"
        }
        return new_id

def save_persons_to_file(persons, lock, filename="persons_detected.txt"):
    """
    Guarda la información de las personas detectadas en un archivo .txt.
    """
    with lock:  # Sincroniza el acceso al diccionario antes de escribir
        with open(filename, "w") as file:
            for person_id, data in persons.items():
                file.write(f"ID: {person_id[:8]}, Edad: {data.get('age', 'N/A')}, Género: {data.get('gender', 'N/A')}, Raza: {data.get('race', 'N/A')}, Emoción: {data.get('emotion', 'N/A')}, Tiempo en pantalla: {data['time_in_screen']} segundos\n")
    print(f"Información guardada en {filename}")

# Definir `stop_threads` como una variable global al inicio
stop_threads = False

def count_time_in_screen(persons, lock):
    """
    Hilo dedicado a contar los segundos que cada persona pasa en pantalla.
    """
    while not stop_threads:
        with lock:
            for person_id, data in persons.items():
                # Incrementar tiempo en pantalla si la persona está presente
                data["time_in_screen"] += 1
        time.sleep(1)  # Esperar 1 segundo antes de la próxima actualización

def analyze_faces_in_realtime():
    global stop_threads  # Declarar como global dentro de la función principal

    # Inicializa la cámara
    #vid = cv2.VideoCapture("rtsp://admin:Admin123.@192.168.1.100:554/profile2/media.smp")
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Diccionario para almacenar datos de las personas detectadas
    persons = {}
    lock = Lock()  # Bloqueo para sincronizar acceso al diccionario

    # Colas para manejar los frames y resultados
    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)

    def capture_frames():
        """
        Captura frames de la cámara y los coloca en una cola.
        """
        while not stop_threads:
            ret, frame = vid.read()
            if not ret:
                print("No se pudo capturar el frame.")
                break
            if not frame_queue.full():
                frame_queue.put(frame)

    def process_frames():
        """
        Procesa frames de la cola y los analiza con DeepFace.
        """
        while not stop_threads:
            if not frame_queue.empty():
                frame = frame_queue.get()
                resized_frame = cv2.resize(frame, (640, 360))
                try:
                    results = DeepFace.analyze(
                        resized_frame,
                        actions=['age', 'gender', 'race', 'emotion'],  # Agregado race y emotion
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    result_queue.put((frame, results))
                except Exception as e:
                    print(f"Error al analizar el rostro: {e}")
                    result_queue.put((frame, None))

    def display_results():
        """
        Extrae resultados de la cola y los muestra en pantalla.
        """
        global stop_threads  # Declarar como global para poder modificarla
        while not stop_threads:
            if not result_queue.empty():
                frame, results = result_queue.get()
                if results:
                    if isinstance(results, list):
                        analyses = results
                    else:
                        analyses = [results]

                    for analysis in analyses:
                        region = analysis.get("region", {})
                        person_id = assign_persistent_id(persons, region, lock)

                        # Actualizar datos de la persona detectada
                        with lock:  # Sincroniza el acceso antes de actualizar
                            persons[person_id]["age"] = analysis.get("age", "N/A")
                            persons[person_id]["gender"] = analysis.get("dominant_gender", "N/A")
                            persons[person_id]["race"] = analysis.get("dominant_race", "N/A")
                            persons[person_id]["emotion"] = analysis.get("dominant_emotion", "N/A")

                        # Dibujar bounding box y atributos
                        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                        if w > 0 and h > 0:
                            cv2.rectangle(frame, (x * 2, y * 2), (x * 2 + w * 2, y * 2 + h * 2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {person_id[:8]}", (x * 2, y * 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f"Edad: {analysis.get('age', 'N/A')}", (x * 2, y * 2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f"Género: {analysis.get('dominant_gender', 'N/A')}", (x * 2, y * 2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f"Raza: {analysis.get('dominant_race', 'N/A')}", (x * 2, y * 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f"Emoción: {analysis.get('dominant_emotion', 'N/A')}", (x * 2, y * 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Mostrar el frame con resultados
                cv2.imshow('Análisis Facial en Tiempo Real', frame)

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads = True
                break

    # Hilos para capturar, procesar, mostrar resultados y contar tiempo
    capture_thread = Thread(target=capture_frames, daemon=True)
    process_thread = Thread(target=process_frames, daemon=True)
    display_thread = Thread(target=display_results, daemon=True)
    time_counter_thread = Thread(target=count_time_in_screen, args=(persons, lock), daemon=True)

    capture_thread.start()
    process_thread.start()
    display_thread.start()
    time_counter_thread.start()

    capture_thread.join()
    process_thread.join()
    display_thread.join()
    time_counter_thread.join()

    vid.release()
    cv2.destroyAllWindows()

    # Guardar datos de las personas detectadas
    save_persons_to_file(persons, lock)

# Ejecutar la función
if __name__ == "__main__":
    analyze_faces_in_realtime()
