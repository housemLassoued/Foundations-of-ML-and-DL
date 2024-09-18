import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# Initialiser le détecteur de visages et le modèle FaceNet
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval()

# Fonction pour extraire les embeddings faciaux
def get_face_embeddings(image_path):
    # Charger l'image
    img = Image.open(image_path)
    
    # Détecter les visages dans l'image
    boxes, probs = mtcnn.detect(img)
    
    if boxes is None:
        return None, None

    # Extraire les visages détectés
    faces = [img.crop(box) for box in boxes]
    
    # Convertir les visages en tenseurs et obtenir les embeddings
    faces = [np.array(face) for face in faces]
    faces = torch.stack([torch.from_numpy(face).float() for face in faces]).permute(0, 3, 1, 2)
    embeddings = model(faces)

    return boxes, embeddings

# Charger les images de référence et obtenir leurs embeddings
reference_image_path = 'reference_face.jpg'
ref_boxes, ref_embeddings = get_face_embeddings(reference_image_path)

# Charger l'image à vérifier et obtenir ses embeddings
unknown_image_path = 'unknown_face.jpg'
unknown_boxes, unknown_embeddings = get_face_embeddings(unknown_image_path)

# Comparer les embeddings des visages
def compare_faces(embeddings1, embeddings2, threshold=0.6):
    if embeddings1 is None or embeddings2 is None:
        return []

    # Calculer les distances entre les embeddings
    distances = torch.cdist(embeddings1, embeddings2)
    matches = distances.min(dim=1)[0] < threshold
    return matches

# Vérifier les visages
if unknown_embeddings is not None:
    matches = compare_faces(ref_embeddings, unknown_embeddings)
    if any(matches):
        print("Personne reconnue !")
    else:
        print("Personne non reconnue.")
else:
    print("Aucun visage détecté dans l'image à vérifier.")
