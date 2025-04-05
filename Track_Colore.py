import cv2
import numpy as np

# Funzione per il tracking del colore
def track_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convertiamo l'immagine in HSV
    mask = cv2.inRange(hsv, lower_color, upper_color)  # Applichiamo la maschera per il colore
    return mask

cap = cv2.VideoCapture(0)

# Colore ROSSO (HSV)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

# Colore BLU (HSV) - definito correttamente per il blu
lower_blue = np.array([100, 150, 0])  # Definizione del blu
upper_blue = np.array([140, 255, 255])

# Colore VERDE (HSV) - definito per il verde
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Colore ROSSO (HSV)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Invertiamo l'immagine per una visione speculare

    # Applichiamo il tracking del colore rosso
    #mask = track_color(frame, lower_red, upper_red)

    # Se vuoi tracciare un altro colore, basta cambiare la linea qui sotto:
    mask = track_color(frame, lower_blue, upper_blue)
    # Per tracciare il blu
    
    # mask = track_color(frame, lower_green, upper_green)
    # Per tracciare il verde

    # Troviamo i contorni dell'oggetto colorato
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Troviamo il contorno più grande
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:  # Soglia per evitare piccoli rumori

            # Calcoliamo il rettangolo di bounding dell'oggetto
            x, y, w, h = cv2.boundingRect(max_contour)
            center = (x + w // 2, y + h // 2)

            # Disegniamo il contorno e il centro dell'oggetto
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 3)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Mostriamo la posizione del centro dell'oggetto
            cv2.putText(frame, f"X: {center[0]} Y: {center[1]}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Calcoliamo il movimento (Delta X, Delta Y)
            if prev_center is not None:
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]

                if abs(dx) > 15 or abs(dy) > 15:
                    direction = "Destra" if dx > 0 else "Sinistra" if dx < 0 else "Giù" if dy > 0 else "Su"
                    cv2.putText(frame, f"Direzione: {direction}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            prev_center = center

    # Mostra l'immagine originale e la maschera
    cv2.imshow("Tracking Colore", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27:  # ESC per uscire
        break

cap.release()
cv2.destroyAllWindows()
