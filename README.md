# README - Détection des Visages avec Dlib et Caffe

## Fonctionnalités
1. **Détection des visages avec Dlib** : Utilise le modèle landmarks68 pour détecter les 68 points clés du visage.
2. **Détection des visages avec Caffe** : Utilise un modèle DNN (Deep Neural Network) pré-entraîné d'OpenCV.


---

## Structure du Projet

### Scripts principaux
- **`image_comparaison.py`** : 
   - Applique les modèles **Dlib** et **Caffe** sur toutes les images d'un dossier.
   - **Note** : Vous devez modifier les chemins (path) des images et des modèles dans le script avant de l'exécuter.

- **`webcam.py`** : 
   - Applique les modèles **Dlib** et **Caffe** au flux vidéo de la webcam en temps réel.

### Scripts individuels par modèle
- **`caffe.py`** : 
   - Applique le modèle **Caffe** uniquement sur le flux vidéo de la webcam.
- **`dlib.py`** : 
   - Applique le modèle **Dlib** uniquement sur le flux vidéo de la webcam.

### Dossier des outils intermédiaires
- **`code_modification_images/`** :
   - Contient tous les scripts utilisés pour modifier les images, notamment :
     - Ajouter du bruit.
     - Appliquer des filtres de couleur.
     - Modifier la luminosité ou le contraste.

---

## Prérequis

### Logiciels
- **Python 3.10**

### Bibliothèques Python
- OpenCV
- Mediapipe (facultatif, si installé par erreur, il n'est pas utilisé ici)
- NumPy
- Dlib
- Matplotlib (si vous souhaitez afficher les images modifiées)

### Modèles nécessaires
- **Dlib** :
  - `models/shape_predictor_68_face_landmarks.dat`
- **Caffe** :
  - `models/deploy.prototxt`
  - `models/res10_300x300_ssd_iter_140000_fp16.caffemodel`

---

## Installation

1. **Cloner le dépôt ou copier les fichiers nécessaires.**
2. Installer les dépendances Python avec la commande :
   ```bash
   pip install opencv-python numpy dlib matplotlib
   ```
3. Vérifier que les fichiers modèles sont présents dans le dossier `models`.

---

## Utilisation

### Script pour le traitement par lot
1. Modifier les chemins d'accès dans le fichier `image_comparaison.py` pour indiquer le dossier contenant les images à analyser.
2. Exécuter le script :
   ```bash
   python image_comparaison.py
   ```

### Script pour la webcam
1. Exécuter le script `webcam.py` :
   ```bash
   python webcam.py
   ```

### Scripts individuels par modèle
- Pour utiliser uniquement **Dlib** :
   ```bash
   python dlib.py
   ```
- Pour utiliser uniquement **Caffe** :
   ```bash
   python caffe.py
   ```

---

## Résultats attendus

### `image_comparaison.py`
- Les résultats des détections (Dlib et Caffe) sont affichés et sauvegardés pour toutes les images d'un dossier.

### `webcam.py`, `caffe.py`, et `dlib.py`
- Détection en temps réel des visages sur le flux vidéo de la webcam :
  - **Caffe** : Affiche des rectangles verts autour des visages détectés.
  - **Dlib** : Affiche des rectangles bleus et les 68 points clés du visage.

---

## Exemple de sortie

### Résultat de la détection sur une image
![Exemple de détection](images/exemple_detection.png)

---

## Auteurs
- **Nom** : Marie HAMADY, Pierre TEIXEIRA
- **Cours** : MTI805 Compréhension de l’image, Hiver 2025
- **Établissement** : École de technologie supérieure (ÉTS)