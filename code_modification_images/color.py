import cv2
import numpy as np
import os

# Charger l'image PNG avec canal alpha
image = cv2.imread("images/imageRef.png", cv2.IMREAD_UNCHANGED)

# Vérifier si l'image a un canal alpha
if image is None:
    print("L'image n'a pas pu être chargée. Vérifiez le chemin du fichier.")
else:
    print("Dimensions de l'image :", image.shape)
    if image.shape[2] == 4:  # RGBA
        # Séparer les canaux B, G, R et Alpha
        b, g, r, alpha = cv2.split(image)
        
        # Créer un masque pour les pixels transparents (alpha == 0)
        transparent_mask = (alpha == 0)  # Pixels où l'alpha est à 0

        # Créer le dossier "couleur" s'il n'existe pas
        output_folder = "images/couleur/input"
        os.makedirs(output_folder, exist_ok=True)

        # Générer 64 couleurs couvrant tout le spectre
        colors = [
            (int(r), int(g), int(b))
            for r in np.linspace(0, 255, 5)  # Rouge : 4 valeurs (0, 85, 170, 255)
            for g in np.linspace(0, 255, 5)  # Vert : 4 valeurs (0, 85, 170, 255)
            for b in np.linspace(0, 255, 5)  # Bleu : 4 valeurs (0, 85, 170, 255)
        ]

    
        # Boucle pour appliquer chaque couleur et sauvegarder l'image
        for i, color in enumerate(colors):
            # Copier les canaux d'origine pour éviter de modifier directement
            b_copy, g_copy, r_copy, alpha_copy = b.copy(), g.copy(), r.copy(), alpha.copy()

            # Appliquer la couleur de remplacement aux pixels transparents
            b_copy[transparent_mask] = color[2]  # Bleu
            g_copy[transparent_mask] = color[1]  # Vert
            r_copy[transparent_mask] = color[0]  # Rouge

            # Rendre ces pixels opaques dans le canal alpha
            alpha_copy[transparent_mask] = 255

            # Fusionner les canaux pour reformer l'image
            image_colored = cv2.merge((b_copy, g_copy, r_copy, alpha_copy))

            # Sauvegarder l'image avec un nom unique
            output_path = os.path.join(output_folder, f"image{i}.png")
            cv2.imwrite(output_path, image_colored)

        print(f"{len(colors)} images sauvegardées dans le dossier '{output_folder}'")
    else:
        print("L'image ne contient pas de canal alpha.")
