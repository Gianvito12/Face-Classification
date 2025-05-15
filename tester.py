import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- Caricamento miglior modello ---
# Carica il miglior modello salvato in 'models/final/best_model.pkl'
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# --- Caricamento dati di test ---
# Carica il dataset di test salvato in 'test/test.pkl'
with open(best_model["test_path"], "rb") as f:
    test_data = np.array(pickle.load(f))

# Separa le caratteristiche (X) e le etichette (y) dai dati di test
X_test = test_data[:, :-1]  
y_test = test_data[:, -1]  
#Controllo se i dati sono scalati o meno
if best_model['scaled']:
    X_test_input=best_model['scaler'].transform(X_test)
else:
    X_test_input=X_test

# --- Predizione e valutazione ---
# Usa il miglior modello per fare predizioni sui dati di test 
y_pred = best_model['model'].predict(X_test_input)

# Calcola l'accuratezza confrontando le etichette previste con quelle reali
accuracy = balanced_accuracy_score(y_test, y_pred)


print(f"== TESTING MIGLIOR MODELLO ==")
print("Accuracy su test set:", accuracy)  # Stampa l'accuratezza sul test set

# --- Confusion matrix ---
# Funzione per calcolare e visualizzare la matrice di confusione
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Greens)  
    plt.title(title)  
    plt.show()  

# Plotta la confusion matrix per il miglior modello sul test set
plot_confusion(y_test, y_pred, f"Confusion Matrix - {best_model['model_name']} (Test Set)")
