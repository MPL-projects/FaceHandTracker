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

    # Créer le dossier "color_filters" s'il n'existe pas
    output_folder = "images/color_filters/input"
    os.makedirs(output_folder, exist_ok=True)

    # Définir les couleurs de filtre (BGR format pour OpenCV)
    color_filters = {
        "red": (0, 0, 255),
        "cyan": (255, 255, 0),
        "green": (0, 255, 0),
        "magenta": (255, 0, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0)
    }

    # Nombre de niveaux d'intensité par couleur
    intensity_steps = 10
    intensity_values = np.linspace(0.1, 1.0, intensity_steps)  # Intensités de 0.1 à 1.0

    # Appliquer les filtres de couleur
    image_index = 1  # Index global pour toutes les images
    for color_name, bgr_filter in color_filters.items():
        for intensity in intensity_values:
            # Créer une matrice de filtre avec l'intensité
            filter_layer = np.full_like(bgr, bgr_filter, dtype=np.uint8)
            blended_image = cv2.addWeighted(bgr, 1 - intensity, filter_layer, intensity, 0)

            # Réassembler avec le canal alpha si présent
            if has_alpha:
                final_image = cv2.merge((blended_image, alpha))
            else:
                final_image = blended_image

            # Ajouter le texte indiquant la couleur et l'intensité
            filter_text = f"{color_name.capitalize()}, Intensity: {intensity:.1f}"
            cv2.putText(
                final_image,
                filter_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            # Sauvegarder l'image avec un nom unique
            output_path = os.path.join(output_folder, f"image_{image_index:03d}.png")
            cv2.imwrite(output_path, final_image)

            # Incrémenter l'index global
            image_index += 1

    print(f"{image_index - 1} images avec différents filtres de couleur ont été sauvegardées dans le dossier '{output_folder}'.")
