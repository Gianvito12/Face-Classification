import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os

# Funzione per plottare la confusion matrix
def plot_confusion(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Greens)
        plt.title(title)
        plt.show()

# Funzione per valutare tutti i modelli in una directory su un validation set
def evaluate_models(model_dir, X_val, y_val, scaler):
    results = {}

    # Caricamento dei modelli non scalati
    with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as f:
        rfc = pickle.load(f)
    with open(os.path.join(model_dir, "logistic_regression.pkl"), "rb") as f:
        lr = pickle.load(f)
    with open(os.path.join(model_dir, "linear_svc.pkl"), "rb") as f:
        lsvc = pickle.load(f)

    # Caricamento dei modelli scalati
    with open(os.path.join(model_dir, "random_forest_scaled.pkl"), "rb") as f:
        rfc_scaled = pickle.load(f)
    with open(os.path.join(model_dir, "logistic_regression_scaled.pkl"), "rb") as f:
        lr_scaled = pickle.load(f)
    with open(os.path.join(model_dir, "linear_svc_scaled.pkl"), "rb") as f:
        lsvc_scaled = pickle.load(f)

    # Valutazione modelli non scalati
    print(f"== Accuracy su dati NON SCALATI ({model_dir}) ==")
    for name, model in [("Random Forest", rfc), ("Logistic Regression", lr), ("Linear SVC", lsvc)]:
        y_pred = model.predict(X_val)
        acc = balanced_accuracy_score(y_val, y_pred)
        print(f"{name}:", acc)
        plot_confusion(y_val, y_pred, f"{name} (no scaling) - {model_dir}")
        results[f"{name} (no scaling)"] = (acc, model, False)

    # Valutazione modelli scalati
    print(f"\n== Accuracy su dati SCALATI ({model_dir}) ==")
    X_val_scaled = scaler.transform(X_val)
    for name, model in [("Random Forest", rfc_scaled), ("Logistic Regression", lr_scaled), ("Linear SVC", lsvc_scaled)]:
        y_pred = model.predict(X_val_scaled)
        acc = balanced_accuracy_score(y_val, y_pred)
        print(f"{name} (scaled):", acc)
        plot_confusion(y_val, y_pred, f"{name} (scaled) - {model_dir}")
        results[f"{name} (scaled)"] = (acc, model, True)

    return results

# Funzione principale di selezione e validazione
def validator():
    # Configurazione per i due tipi di dataset: uniform e default
    configs = {
        "uniform": {
            "val_path": "validation/uniform/validation.pkl",
            "test_path": "test/uniform/test.pkl",
            "model_dir": "models/uniform/"
        },
        "default": {
            "val_path": "validation/default/validation.pkl",
            "test_path": "test/default/test.pkl",
            "model_dir": "models/default/"
        }
    }

    all_results = {}

    # Cicla su entrambe le configurazioni (uniform e default)
    for key, cfg in configs.items():
        # Caricamento del validation set
        with open(cfg["val_path"], "rb") as f:
            val_data = np.array(pickle.load(f))
        X_val = val_data[:, :-1]
        y_val = val_data[:, -1]

        # Caricamento dello scaler associato
        with open(os.path.join(cfg["model_dir"], "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        # Valutazione dei modelli su validation set
        print(f"\n==== VALUTAZIONE MODELLI: {key.upper()} ====")
        results = evaluate_models(cfg["model_dir"], X_val, y_val, scaler)

        # Selezione del miglior modello per questa configurazione
        best_model_name = max(results, key=lambda k: results[k][0])
        all_results[key] = {
            "model_name": best_model_name,
            "accuracy": results[best_model_name][0],
            "model": results[best_model_name][1],
            "scaled": results[best_model_name][2],
            "scaler": scaler if results[best_model_name][2] else None,
            "test_path": cfg["test_path"]
        }

    # Confronta il miglior modello tra uniform e default
    best_config = max(all_results, key=lambda k: all_results[k]["accuracy"])
    best = all_results[best_config]

    print(f"\n== MIGLIOR MODELLO FINALE ({best_config.upper()}) ==")
    print(f"{best['model_name']} con accuracy validazione = {best['accuracy']:.4f}")

    with open("best_model.pkl", "wb") as f:
        pickle.dump(best, f)

    
def main():
    validator()

if __name__ == "__main__":
    main()
