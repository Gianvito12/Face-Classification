import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

def learning(isUniform):
    if isUniform:
        path = "train/uniform/train.pkl"
        model_dir = "models/uniform/"
    else:
        path = "train/default/train.pkl"
        model_dir = "models/default/"

    os.makedirs(model_dir, exist_ok=True)  # Crea la directory se non esiste

    # Caricamento dei dati di addestramento
    with open(path, "rb") as f:
        data = np.array(pickle.load(f))

    # Separa le caratteristiche e le etichette
    X = data[:, :-1]
    y = data[:, -1]

    # Creazione e addestramento dei modelli NON scalati
    rfc = RandomForestClassifier()
    rfc.fit(X, y)

    lr = LogisticRegression()
    lr.fit(X, y)

    lsvc = LinearSVC()
    lsvc.fit(X, y)

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # Addestramento dei modelli scalati
    rfc_scaled = RandomForestClassifier()
    rfc_scaled.fit(X_scaled, y)

    lr_scaled = LogisticRegression()
    lr_scaled.fit(X_scaled, y)

    lsvc_scaled = LinearSVC()
    lsvc_scaled.fit(X_scaled, y)

    # Salvataggio dei modelli NON scalati
    with open(os.path.join(model_dir, "random_forest.pkl"), "wb") as f:
        pickle.dump(rfc, f)

    with open(os.path.join(model_dir, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(lr, f)

    with open(os.path.join(model_dir, "linear_svc.pkl"), "wb") as f:
        pickle.dump(lsvc, f)

    # Salvataggio dei modelli SCALATI
    with open(os.path.join(model_dir, "random_forest_scaled.pkl"), "wb") as f:
        pickle.dump(rfc_scaled, f)

    with open(os.path.join(model_dir, "logistic_regression_scaled.pkl"), "wb") as f:
        pickle.dump(lr_scaled, f)

    with open(os.path.join(model_dir, "linear_svc_scaled.pkl"), "wb") as f:
        pickle.dump(lsvc_scaled, f)

    # Salvataggio dello scaler
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

def main():
    print("Addestramento default...")
    learning(False)
    print("Addestramento uniform...")
    learning(True)

if __name__ == "__main__":
    main()
