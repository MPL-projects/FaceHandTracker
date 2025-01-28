import cv2
import dlib
import os

# Initialiser les modèles
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

opencv_dnn_model = cv2.dnn.readNetFromCaffe(
    prototxt="models/deploy.prototxt",
    caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Détection des visages avec OpenCV DNN
def cv_dnn_detect_faces(image, model, min_confidence=0.5):
    image_height, image_width, _ = image.shape
    preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    model.setInput(preprocessed_image)
    results = model.forward()
    faces = []

    for face in results[0][0]:
        confidence = face[2]
        if confidence > min_confidence:
            bbox = face[3:]
            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)
            faces.append((x1, y1, x2, y2))

    return faces


# Fusion des résultats Dlib et OpenCV DNN
def detect_and_annotate(image_path, output_folder):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image: {image_path}")
        return

    # Créer une copie pour affichage
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détection avec OpenCV DNN
    dnn_faces = cv_dnn_detect_faces(image, opencv_dnn_model)

    # Dessiner les résultats OpenCV DNN
    for (x1, y1, x2, y2) in dnn_faces:
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 20)
        # cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 20)

    # Détection avec Dlib
    dlib_faces = dlib_detector(gray)

    for face in dlib_faces:
        # Dessiner un rectangle autour du visage détecté
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 0), 20)
        # cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 20)


    # Sauvegarder le résultat
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_file, output_image)



def detect_faces_from_webcam():
    # Initialiser la capture de la webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Detection de visages")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Erreur")
            break

        frame = cv2.flip(frame, 1)  # Inverser horizontalement pour un effet miroir

        # Conversion de l'image en niveaux de gris pour Dlib
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection des visages avec Dlib
        dlib_faces = dlib_detector(gray_frame)

        # Dessiner les rectangles pour chaque visage détecté
        for face in dlib_faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 10)

        # Détection des visages avec OpenCV DNN
        dnn_faces = cv_dnn_detect_faces(frame, opencv_dnn_model)

        # Dessiner les résultats OpenCV DNN
        for (x1, y1, x2, y2) in dnn_faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 10)

        # Afficher la vidéo avec les visages détectés
        cv2.imshow("Detection de visages", frame)

        # Quitter si la touche 'ESC' est pressée
        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("ESC")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_faces_from_webcam()
