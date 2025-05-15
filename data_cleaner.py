import cv2
import os
import numpy as np

# Carica il classificatore Haar per il rilevamento facciale
cascadeClassifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

def scale_crop_discard(img_path):
    # Legge l'immagine
    img = cv2.imread(img_path)
    
    # Converte l'immagine in scala di grigi per il rilevamento facciale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Rileva i volti nell'immagine
    faces = cascadeClassifier.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=9)
    
    # Scarta l'immagine se non viene trovato esattamente un volto
    if len(faces) != 1:
        return None
    
    # Estrai coordinate e dimensioni del volto rilevato
    x, y, w, h = faces[0]
    
    # Calcola il centro del volto e il centro dell'immagine
    height, width = img.shape[:2]
    face_center = np.array([x + w / 2, y + h / 2])
    img_center = np.array([width / 2, height / 2])
    
    # Calcola la traslazione necessaria per centrare il volto
    translation = img_center - face_center
    translation_matrix = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]]
    ], dtype=np.float32)
    
    # Applica la traslazione per centrare il volto
    img_aligned = cv2.warpAffine(img, translation_matrix, (width, height))
    
    # Calcola un fattore di scala (zoom) e crea la matrice di trasformazione
    scale = 0.85 * img.shape[1] / w
    zoom_matrix1 = cv2.getRotationMatrix2D(face_center, 0, scale)
    
    # Applica lo zoom (ridimensionamento centrato sul volto)
    img_cropped = cv2.warpAffine(img_aligned, zoom_matrix1, (width, height))
    
    # Ritorna l'immagine elaborata
    return img_cropped

def main():
    current_dir = os.getcwd() 

    # Percorso di destinazione dove salvare le immagini croppate
    destination_dir = os.path.join(current_dir, "cropped_real")

    # Percorso delle immagini non croppate
    current_dir = os.path.join(current_dir, "uncropped_real")

    # Ottiene la lista dei soggetti (cartelle) all'interno di uncropped_real
    subjects = [subject for subject in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, subject))]

    for subject in subjects:
        current_subject = os.path.join(current_dir, subject)

        # Ottiene la lista delle immagini del soggetto
        images = [image for image in os.listdir(current_subject) if os.path.isfile(os.path.join(current_subject, image))]

        # Crea la directory di destinazione per il soggetto
        current_subject_destination_path = os.path.join(destination_dir, subject)
        os.makedirs(current_subject_destination_path,exist_ok=True)

        for image in images:
            current_img = os.path.join(current_subject, image)

            # Elabora l'immagine 
            result = scale_crop_discard(current_img)

            # Se non è stato rilevato un volto valido, passa oltre
            if result is None:
                continue

            # Costruisce il percorso per salvare l'immagine elaborata
            current_image_destination_path = os.path.join(current_subject_destination_path, image)

            # Salva l'immagine elaborata
            cv2.imwrite(current_image_destination_path, result)

        print(f"{subject} terminato") 

if __name__ == "__main__":
    main()  