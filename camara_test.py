import cv2

def get_camera():
    # Probar índices locales (cámaras físicas)
    for index in range(3):  # Prueba los índices 0, 1, 2
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Cámara detectada con índice {index}.")
            return cap
        cap.release()

    # Probar una cámara RTSP (IP)
    rtsp_url = "rtsp://admin:Admin123.@192.168.1.100:554/profile2/media.smp"
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        print("Cámara RTSP detectada correctamente.")
        return cap

    # Si no se encuentra ninguna cámara
    print("No se pudo detectar ninguna cámara.")
    return None

# Obtener cámara
vid = get_camera()
if vid is None:
    exit()

# Leer frames
while True:
    ret, frame = vid.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break
    cv2.imshow("Cámara", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
        break

vid.release()
cv2.destroyAllWindows()
