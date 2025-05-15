import pickle
import skimage.feature
import os
import numpy as np
import cv2
import random

def creator(current_path, dir,isUniform):
    # Determina il percorso della directory corrente
    current_dir = os.path.join(current_path, dir)
    if isUniform:
        os.makedirs(os.path.join(current_dir,"uniform"),exist_ok=True)
    else:
        os.makedirs(os.path.join(current_dir,"default"),exist_ok=True)
    
    # Lista per raccogliere i dati (vettori di caratteristiche)
    data = []
    total_images = 0  

    # Itera su ciascuna delle classi ('real' e 'fake')
    for label in ['real', 'fake']:
        print(f"\n[INFO] Inizio elaborazione classe '{label}'")

        # Ottiene la lista dei soggetti per la classe corrente
        subjects = [
            subject for subject in os.listdir(os.path.join(current_dir, label))
            if os.path.isdir(os.path.join(current_dir, label, subject))
        ]

        # Itera sui soggetti (individui) per ogni classe
        for subject in subjects:
            current_subject = os.path.join(current_dir, label, subject)

            # Ottiene la lista delle immagini per il soggetto corrente
            images = [
                image for image in os.listdir(current_subject)
                if os.path.isfile(os.path.join(current_subject, image))
            ]

            # Processa ogni immagine
            for img in images:
                current_img = os.path.join(current_subject, img)
                mat_img = cv2.imread(current_img)
                
                # Se l'immagine non è valida, salta e continua con quella successiva
                if mat_img is None:
                    continue
                
                # Converte l'immagine in scala di grigi
                mat_img_gray = cv2.cvtColor(mat_img, cv2.COLOR_BGR2GRAY)

                # Calcola il Local Binary Pattern (LBP) con metodo passato per parametro
                if isUniform:
                    lbp_current = skimage.feature.local_binary_pattern(mat_img_gray, P=8, R=1.0, method='uniform')
                    hist, _ = np.histogram(lbp_current,bins=10,density=True)
                else:
                    lbp_current = skimage.feature.local_binary_pattern(mat_img_gray, P=256, R=1.0, method='default')
                    hist, _ = np.histogram(lbp_current,bins=256,density=True)

                # Etichetta: 1 per 'fake', 0 per 'real'
                label_value = 1 if label == 'fake' else 0
                
                # Aggiunge l'etichetta al vettore dell'istogramma
                hist_with_label = np.append(hist, label_value)
                data.append(hist_with_label)
                
                total_images += 1
                # Ogni 100 immagini, stampa il progresso
                if total_images % 100 == 0:
                    print(f"[PROGRESS] {total_images} immagini elaborate finora...") 
            
            
    # Mescola i dati 
    random.shuffle(data)
    #Determina dove salvare 
    if isUniform:
        path="uniform"
    else:
        path="default"
    # Salva i dati in un file 
    output_file = os.path.join(current_dir,path, f"{dir}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

def main():
    # Ottieni il percorso corrente
    current_path = os.getcwd()
    # Esegui la funzione di creazione dei dati per 'train', 'validation', e 'test'
    for dir in ['train', 'validation', 'test']:
        creator(current_path, dir,False)
        creator(current_path, dir,True)


if __name__ == "__main__":
    main()
