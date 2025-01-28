import cv2
import numpy as np
import os

# Charger l'image PNG avec canal alpha
image = cv2.imread("images/example1.jpg", cv2.IMREAD_UNCHANGED)

# Vérifier si l'image est correctement chargée
if image is None:
    print("Erreur : L'image n'a pas pu être chargée. Vérifiez le chemin du fichier.")
else:
    print("Dimensions de l'image :", image.shape)

    # Vérifier si l'image a un canal alpha
    has_alpha = image.shape[2] == 4  # Vérifie si RGBA

    # Si l'image a un canal alpha, séparer le canal Alpha
    if has_alpha:
        bgr = image[:, :, :3]  # Extraire les canaux BGR
        alpha = image[:, :, 3]  # Extraire le canal alpha
    else:
        bgr = image
        alpha = None

    # Convertir l'image BGR en HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Créer le dossier "brightness" s'il n'existe pas
    output_folder = "images/brightness/input"
    os.makedirs(output_folder, exist_ok=True)

    # Générer 100 niveaux de luminosité (de 0 à 255 pour inclure les extrêmes)
    brightness_values = np.linspace(0, 255, 100, dtype=np.uint8)

    # Boucle pour appliquer chaque niveau de luminosité et sauvegarder l'image
    for i, brightness in enumerate(brightness_values):
        hsv_copy = hsv.copy()

        # Ajuster la composante V (Valeur) dans l'espace HSV
        hsv_copy[:, :, 2] = np.clip(hsv_copy[:, :, 2] * (brightness / 128), 0, 255).astype(np.uint8)

        # Convertir l'image HSV modifiée en BGR
        bgr_brightness = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR)

        # Réassembler avec le canal alpha si présent
        if has_alpha:
            image_brightness = cv2.merge((bgr_brightness, alpha))
        else:
            image_brightness = bgr_brightness

   

        # Sauvegarder l'image avec un nom unique
        output_path = os.path.join(output_folder, f"image_brightness_{i}.png")
        cv2.imwrite(output_path, image_brightness)

    print(f"{len(brightness_values)} images avec différentes luminosités ont été sauvegardées dans le dossier '{output_folder}'.")
