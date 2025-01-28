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

    # Créer le dossier "contrast" s'il n'existe pas
    output_folder = "images/contrast/input"
    os.makedirs(output_folder, exist_ok=True)

    # Générer 100 niveaux de contraste (de 0.5 à 2.0)
    contrast_values = np.linspace(0.5, 2.0, 100)

    for i, contrast in enumerate(contrast_values):
        # Appliquer le contraste
        contrast_image = cv2.convertScaleAbs(bgr, alpha=contrast, beta=0)

        # Réassembler avec le canal alpha si présent
        if has_alpha:
            final_image = cv2.merge((contrast_image, alpha))
        else:
            final_image = contrast_image

        # Ajouter le texte indiquant le niveau de contraste
        contrast_text = f"Contrast: {contrast:.2f}"
        cv2.putText(
            final_image,
            contrast_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # Sauvegarder l'image
        output_path = os.path.join(output_folder, f"image_contrast_{i+1:03d}.png")
        cv2.imwrite(output_path, final_image)

    print(f"{len(contrast_values)} images avec différents niveaux de contraste ont été sauvegardées dans le dossier '{output_folder}'.")
