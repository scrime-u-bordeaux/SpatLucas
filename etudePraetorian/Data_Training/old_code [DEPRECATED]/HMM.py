import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# ==============================
# 1. Entraînement du modèle HMM
# ==============================
def train_hmm(data_paths, numeric_columns, n_components=12, n_iter=300, random_state=42):
    all_sequences = []
    lengths = []
    for path in data_paths:
        df = pd.read_csv(path)
        X = df[numeric_columns].values
        print("Données lues :", X.shape, "extraits de", path)
        print(X)
        all_sequences.append(X)
        lengths.append(len(X))
        print("Utilisé pour l'entrainement : ", path)
    print("features utilisées :", numeric_columns)

    X_all = np.vstack(all_sequences)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(X_scaled, lengths)

    # On stocke les colonnes dans l'objet modèle pour la prédiction
    model.feature_names = numeric_columns  

    print("*Modèle entraîné ! Score log-vraisemblance :", model.score(X_scaled))
    return model, scaler


# ==============================
# 2. Génération de trajectoires
# ==============================
def generate_trajectory(model, scaler, n_samples=300):
    X_gen_scaled, states = model.sample(n_samples)
    X_gen = scaler.inverse_transform(X_gen_scaled)

    x_gen = X_gen[:, -2]
    y_gen = X_gen[:, -1]

    plt.figure(figsize=(6, 6))
    plt.plot(x_gen, y_gen, marker="o", markersize=2, linestyle="-", alpha=0.7)
    plt.title("Trajectoire générée par HMM")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

    return X_gen, states


# ==============================
# 3. Spatialisation à partir audio
# ==============================
def predict_spatialisation(model, scaler, df):
    numeric_columns = model.feature_names  # colonnes utilisées à l'entraînement
    n = len(df)

    # On prend seulement les colonnes utiles
    missing = [c for c in numeric_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans df : {missing}")

    X_incomplete = df[numeric_columns].fillna(0.0).values  # NaN remplacés par 0

    # Normalisation
    X_incomplete_scaled = scaler.transform(X_incomplete)

    # Prédiction des états
    hidden_states = model.predict(X_incomplete_scaled)

    # Associer chaque état à ses coordonnées moyennes apprises
    means = scaler.inverse_transform(model.means_)
    print("Moyennes des états (dénormalisées) :\n", means)
    coords_means = means[:, -2:]  # les 2 dernières colonnes sont x/y

    coords_pred = np.array([coords_means[state] for state in hidden_states])

    return coords_pred, hidden_states


import os
import glob
# ==============================
# 4. Exemple d’utilisation
# ==============================
if __name__ == "__main__":
    # ---- Choix des paramètres ----
    train_seqs = [0,1,2,3]   # numéros de séquences pour l’entraînement
    data_dir = "resampled_results2"
    out_dir = "sequences_datasets_predicted"
    os.makedirs(out_dir, exist_ok=True)

    # Récupération des fichiers d’entraînement
    DATA_TRAIN_PATHS = []
    for seq in train_seqs:
        pattern = os.path.join(data_dir, f"dataset_track*_seq{seq}.csv")
        for file_path in glob.glob(pattern):
            if "_no_coord" not in file_path:
                DATA_TRAIN_PATHS.append(file_path)

    # Récupération des fichiers sans coordonnées
    DATA_PREDICT_PATHS = glob.glob(os.path.join(data_dir, "*_no_coord.csv"))

    numeric_columns = [
        "rms_Sample",
        "rms_Voc 1",
        "rms_Voc 2",
        "rms_Guitare",
        "rms_Basse",
        "rms_BatterieG",
        "rms_BatterieD",
        "beat", 
        "measure", 
        "x_Voc 1", 
        "y_Voc 1"
        ]

    n_components = 200 # nombre d'états cachés
    n_iter = 1000       # itérations max d’entrainement
    n_samples = 80     # nombre de points générés pour trajectoire

    # ---- Entraînement ----
    model, scaler = train_hmm(DATA_TRAIN_PATHS, numeric_columns,
                              n_components=n_components, n_iter=n_iter)

    # ---- Génération d’une trajectoire synthétique ----
    X_gen, states = generate_trajectory(model, scaler, n_samples=n_samples)

    instr_map = {"Sample" : 0, "Voc 1": 1, "Voc 2": 2, "Guitare": 3, "Basse" : 4, "BatterieG" : 5, "BatterieD" : 6}

    import re
    # ---- Prédiction des coordonnées pour tous les fichiers no_coord ----


    from collections import defaultdict

predicted_csv_paths = []
tracks_grouped = defaultdict(list)  # dictionnaire {track_id: [liste_csv]}

for file_path in DATA_PREDICT_PATHS:
    print(f"\nPrédiction pour : {file_path}")
    df_test = pd.read_csv(file_path)
    
    coords_pred, states = predict_spatialisation(model, scaler, df_test)

    # Récupération de l’ID de la track et de la séquence
    m = re.search(r"track(\d+)_seq(\d+)", file_path)
    track_id = int(m.group(1)) if m else -1
    seq_id = int(m.group(2)) if m else -1

    # Détection automatique de l’instrument cible via les colonnes x_ / y_ vides
    target_cols_x = [c for c in df_test.columns if c.startswith("x_") and df_test[c].isnull().all()]
    target_cols_y = [c for c in df_test.columns if c.startswith("y_") and df_test[c].isnull().all()]
    if target_cols_x and target_cols_y:
        instr_name = target_cols_x[0][2:]  # enlève le "x_"
        instr_id = instr_map.get(instr_name, 0)
    else:
        instr_name = "unknown"
        instr_id = 0

    # Remplissage des colonnes avec les coordonnées prédites
    df_test[f"x_{instr_name}"] = [c[0] for c in coords_pred]
    df_test[f"y_{instr_name}"] = [c[1] for c in coords_pred]

    # Nouveau chemin dans le dossier de sortie
    base_name = os.path.basename(file_path).replace("_no_coord.csv", "_predicted_coord.csv")
    out_path = os.path.join(out_dir, base_name)

    df_test.to_csv(out_path, index=False)
    print(f"Coordonnées prédites enregistrées dans : {out_path}")
    print(f"Track ID = {track_id}, Seq ID = {seq_id}, Instrument = {instr_name}")

    # Ajouter aux groupes par track
    tracks_grouped[track_id].append(out_path)

from convert_to_max_data import convert_to_max_data  

RESULT_FOLDER_MAX = "max_data_predicted"
os.makedirs(RESULT_FOLDER_MAX, exist_ok=True)

# Supprimer le fichier seq<instr_id>.txt avant conversion
seq_file = os.path.join(RESULT_FOLDER_MAX, f"seq{instr_id + 1}.txt")
if os.path.exists(seq_file):
    os.remove(seq_file)
for track_id, csv_files in tracks_grouped.items():
    print(f"\nConversion Max pour track {track_id} avec {len(csv_files)} fichiers...")
    convert_to_max_data(csv_files=csv_files, track_id=track_id, instr_coord_id=instr_id)

