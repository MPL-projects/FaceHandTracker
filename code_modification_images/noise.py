import os
import cv2
import numpy as np

def add_noise(image, noise_type="gaussian", intensity=0.1):
    """
    Ajoute du bruit à une image.
    :param image: Image source (format numpy array).
    :param noise_type: Type de bruit ('gaussian' ou 'salt_pepper').
    :param intensity: Intensité du bruit (0.0 à 1.0 pour salt_pepper, écart-type pour gaussian).
    :return: Image bruitée.
    """
    if noise_type == "gaussian":
        mean = 0
        stddev = intensity * 255  # Intensité proportionnelle à l'écart-type
        noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    elif noise_type == "salt_pepper":
        noisy_image = image.copy()
        prob = intensity  # Proportion de pixels affectés
        salt_prob = prob / 2

        # Ajouter des points "sel"
        num_salt = int(salt_prob * image.size)
        coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[coords_salt[0], coords_salt[1]] = 255

        # Ajouter des points "poivre"
        num_pepper = int((prob - salt_prob) * image.size)
        coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[coords_pepper[0], coords_pepper[1]] = 0

        return noisy_image
    else:
        raise ValueError(f"Type de bruit non pris en charge : {noise_type}")

def generate_noisy_images(input_image_path, gaussian_folder, salt_pepper_folder, num_images=100):
    """
    Génère une série d'images avec bruit croissant et affiche l'intensité du bruit sur l'image.
    :param input_image_path: Chemin de l'image source.
    :param gaussian_folder: Dossier de sortie pour le bruit gaussien.
    :param salt_pepper_folder: Dossier de sortie pour le bruit salt & pepper.
    :param num_images: Nombre d'images à générer.
    """
    # Charger l'image source
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {input_image_path}")

    # Créer les dossiers de sortie
    os.makedirs(gaussian_folder, exist_ok=True)
    os.makedirs(salt_pepper_folder, exist_ok=True)

    # Générer les images avec du bruit croissant
    for i in range(num_images):
        intensity = i / (num_images - 1)  # Intensité croissante de 0 à 1
        intensity_text = f"Noise : {intensity:.2f}"
        intensity_text_gauss = f"Noise : {i} / 100"

        # Bruit gaussien
        noisy_gaussian = add_noise(image, noise_type="gaussian", intensity=intensity)
        cv2.putText(noisy_gaussian, intensity_text_gauss, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        gaussian_output_path = os.path.join(gaussian_folder, f"image_gaussian_{i+1:03d}.png")
        cv2.imwrite(gaussian_output_path, noisy_gaussian)

        # Bruit salt & pepper
        noisy_salt_pepper = add_noise(image, noise_type="salt_pepper", intensity=intensity * 0.1)
        cv2.putText(noisy_salt_pepper, intensity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        salt_pepper_output_path = os.path.join(salt_pepper_folder, f"image_salt_pepper_{i+1:03d}.png")
        cv2.imwrite(salt_pepper_output_path, noisy_salt_pepper)

        print(f"Image {i+1}/{num_images} générée.")

# Paramètres
input_image_path = "images/example1.jpg"
gaussian_folder = "images/noise/gaussian/input"
salt_pepper_folder = "images/noise/salt_pepper/input"

# Génération des images bruitées
generate_noisy_images(input_image_path, gaussian_folder, salt_pepper_folder, num_images=100)
