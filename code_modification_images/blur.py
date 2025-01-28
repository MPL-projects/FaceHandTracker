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

    # Séparer le canal alpha si présent
    if has_alpha:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        bgr = image
        alpha = None

    # Créer le dossier "blur" s'il n'existe pas
    output_folder = "images/blur/input"
    os.makedirs(output_folder, exist_ok=True)

    # Générer 100 niveaux de flou (noyaux impairs croissants)
    blur_values = range(1, 200, 2)  # 100 valeurs de 1 à 199

    for i, blur in enumerate(blur_values):
        # Appliquer un flou gaussien avec un noyau de taille blur x blur
        blurred_image = cv2.GaussianBlur(bgr, (blur, blur), 0)

        # Réassembler avec le canal alpha si présent
        if has_alpha:
            final_image = cv2.merge((blurred_image, alpha))
        else:
            final_image = blurred_image

        # Ajouter le texte indiquant le niveau de flou
        blur_text = f"Blur: {blur}"
        cv2.putText(
            final_image,
            blur_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # Sauvegarder l'image
        output_path = os.path.join(output_folder, f"image_blur_{i+1:03d}.png")
        cv2.imwrite(output_path, final_image)

    print(f"{len(blur_values)} images avec différents niveaux de flou ont été sauvegardées dans le dossier '{output_folder}'.")
