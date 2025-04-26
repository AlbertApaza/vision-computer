import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from ultralytics import YOLO
import mss
import pyautogui

# ====== Cargar Modelo YOLOv8 entrenado ======
model = YOLO("best.pt")  # Puedes cambiar a 'yolov8s.pt' para más precisión

# ====== Función para capturar la pantalla ======
def captura_pantalla():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Captura la pantalla completa
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)  # Convertir a numpy array
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convertir a formato OpenCV
        return img

# ====== Configurar Tkinter para selección ======
root = tk.Tk()
root.withdraw()

opcion = simpledialog.askstring("Modo de Monitoreo", "Elige un modo:\n1 - Video (archivo)\n2 - Pantalla en vivo")

if opcion == "1":
    # Selección de archivo de video
    video_path = filedialog.askopenfilename(title="Selecciona un video",
                                            filetypes=[("Archivos de video", "*.mp4;*.avi;*.mov;*.mkv")])
    if not video_path:
        messagebox.showerror("Error", "No seleccionaste un archivo. Saliendo...")
        exit()
    cap = cv2.VideoCapture(video_path)

elif opcion == "2":
    cap = None  # Indicamos que usaremos captura de pantalla

else:
    messagebox.showerror("Error", "Opción inválida. Saliendo...")
    exit()

# Sustraedor de fondo para detectar cambios en la escena
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    if cap:
        ret, frame = cap.read()
        if not ret:
            break  # Salir si el video terminó
    else:
        frame = captura_pantalla()  # Capturar la pantalla en tiempo real

    # ====== DETECCIÓN DE MOVIMIENTO ======
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar ruido
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ====== DETECCIÓN DE OBJETOS Y PERSONAS CON YOLO ======
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]  # Nombre del objeto detectado
            confianza = float(box.conf[0])

            if confianza > 0.5:  # Filtramos detecciones poco seguras
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {confianza:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Mostrar resultado
    cv2.imshow("Monitoreo de Seguridad", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Presiona 'q' para salir
        break

if cap:
    cap.release()
cv2.destroyAllWindows()
