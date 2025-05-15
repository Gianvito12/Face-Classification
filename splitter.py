import os
import random
import shutil

def splitter(current_path, dir):
    current_dir = os.path.join(current_path, dir)  # Costruisce il percorso completo della directory da processare

    # Crea una lista di tutte le sottocartelle (soggetti) presenti nella directory corrente
    subjects = [subject for subject in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, subject))]
    
    #Randomizzo l'ordine dei soggetti
    random.seed(42)
    random.shuffle(subjects)

    # Calcola il numero di campioni per training e validation (il resto andrà al test)
    train_samples_len = int(len(subjects) * 0.6)
    validation_samples_len = int(len(subjects) * 0.2)

    # Seleziona casualmente i soggetti per il training set
    train_samples = random.sample(subjects, train_samples_len)

    # Ottiene i soggetti rimanenti non usati per il training
    remaining_subjects = [subject for subject in subjects if subject not in train_samples]

    # Seleziona casualmente i soggetti per il validation set
    validation_samples = random.sample(remaining_subjects, validation_samples_len)

    # I soggetti rimanenti vanno nel test set
    test_samples = [subject for subject in remaining_subjects if subject not in validation_samples]

    # Determina se la directory corrente è "real" o "fake"
    if "real" in dir:
        name_dir = "real"
    else:
        name_dir = "fake"

    # Crea i percorsi per le cartelle train, validation e test
    train_dir = os.path.join(current_path, "train")
    test_dir = os.path.join(current_path, "test")
    validation_dir = os.path.join(current_path, "validation")

    # Crea le sottocartelle per "real" o "fake" all'interno di train, validation e test
    for dir in [train_dir, validation_dir, test_dir]:
        os.makedirs(os.path.join(dir, name_dir),exist_ok=True)

    # Copia i campioni del training set
    for i, sample in enumerate(train_samples):
        source_path = os.path.join(current_dir, sample)
        dest_path = os.path.join(train_dir, name_dir, sample)
        try:
            shutil.copytree(source_path, dest_path)  # Copia l'intera cartella del soggetto
            percent_complete = (i + 1) / train_samples_len * 100  # Calcola percentuale completata
            print(f"Copia completata: {percent_complete:.2f}% train {name_dir}")  
        except Exception as e:
            print(f"Errore durante la copia di {sample}: {e}")  

    # Copia i campioni del validation set
    for i, sample in enumerate(validation_samples):
        source_path = os.path.join(current_dir, sample)
        dest_path = os.path.join(validation_dir, name_dir, sample)
        try:
            shutil.copytree(source_path, dest_path)
            percent_complete = (i + 1) / validation_samples_len * 100
            print(f"Copia completata: {percent_complete:.2f}% validation {name_dir}")
        except Exception as e:
            print(f"Errore durante la copia di {sample}: {e}")

    # Copia i campioni del test set
    total_test_samples = len(test_samples)
    for i, sample in enumerate(test_samples):
        source_path = os.path.join(current_dir, sample)
        dest_path = os.path.join(test_dir, name_dir, sample)
        try:
            shutil.copytree(source_path, dest_path)
            percent_complete = (i + 1) / total_test_samples * 100
            print(f"Copia completata: {percent_complete:.2f}% test {name_dir}")
        except Exception as e:
            print(f"Errore durante la copia di {sample}: {e}")

def main():
    current_path = os.getcwd()  # Ottiene il percorso corrente di esecuzione
    #Controllo se sono presenti le cartelle
    for path in ["train","validation","test"]:
        path_to_check=os.path.join(current_path, path)
        if os.path.exists(path_to_check):
            try:
                shutil.rmtree(path_to_check)
                print(f"Eliminata la cartella: {path_to_check}")
            except OSError as e:
                print(f"Errore durante l'eliminazione di {path_to_check}: {e}")
    dirs = ["cropped_fakes", "cropped_real"]  # Elenco delle directory da processare
    for dir in dirs:
        splitter(current_path, dir)  # Chiama la funzione splitter per ciascuna directory

if __name__ == "__main__":
    main()  # Esegue la funzione main solo se lo script è eseguito direttamente
