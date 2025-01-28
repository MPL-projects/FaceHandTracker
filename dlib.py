import cv2
import dlib

# Charger le détecteur de visages de Dlib
detector = dlib.get_frontal_face_detector()

# Charger le prédicteur de points clés du visage (68 points clés)
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur: Impossible d'accéder à la webcam.")
    exit()
while True:
    # Lire une frame de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = detector(gray)

    for face in faces:
        # Dessiner un rectangle autour du visage
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extraire les points clés du visage
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Afficher le résultat
    cv2.imshow("Suivi facial avec Dlib", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()