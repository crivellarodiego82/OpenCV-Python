import cv2

# Carica il classificatore Haar per il volto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    direction = ""

    for (x, y, w, h) in faces:
        # Disegna rettangolo sul volto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calcola centro del volto
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Calcola direzione
        if prev_center is not None:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]

            if abs(dx) > 15 or abs(dy) > 15:
                if abs(dx) > abs(dy):
                    direction = "Destra" if dx > 0 else "Sinistra"
                else:
                    direction = "Giù" if dy > 0 else "Su"

        prev_center = (cx, cy)
        break  # Considera solo il primo volto rilevato

    if direction:
        cv2.putText(frame, f"Direzione volto: {direction}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Direction Tracker", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
