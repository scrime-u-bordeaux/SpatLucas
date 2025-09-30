import os
import glob
import re
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. Dataset PyTorch
# ==============================
class SequenceDataset(Dataset):
    def __init__(self, data, targets, seq_len=50):
        self.data = data
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.seq_len]
        y = self.targets[idx:idx+self.seq_len]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ==============================
# 2. Modèle LSTM
# ==============================
class SpatialLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SpatialLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# ==============================
# 3. Entraînement du modèle
# ==============================
def train_lstm(train_loader, input_size, output_size, hidden_size=64, num_layers=2, lr=1e-3, n_epochs=20):
    model = SpatialLSTM(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss={total_loss/len(train_loader):.4f}")

    return model

# ==============================
# 4. Fonction principale
# ==============================
def run_training(data_paths, numeric_columns, coord_columns=["x_Voc 1", "y_Voc 1"],
                 seq_len=50, batch_size=32, n_epochs=20):
    
    # --- Lire et concaténer les données
    all_data = []
    for path in data_paths:
        df = pd.read_csv(path)
        X = df[numeric_columns].values
        all_data.append(X)

    X_all = np.vstack(all_data)

    # --- Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # --- Features et targets
    X_features = X_scaled[:, :-2]  # toutes les colonnes sauf x,y
    Y_targets = X_scaled[:, -2:]   # seulement x,y

    # --- Dataset/Dataloader
    dataset = SequenceDataset(X_features, Y_targets, seq_len=seq_len)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Entraînement
    model = train_lstm(train_loader, input_size=X_features.shape[1], output_size=Y_targets.shape[1],
                       hidden_size=128, num_layers=2, lr=1e-3, n_epochs=n_epochs)

    return model, scaler

# ==============================
# 5. Prédiction pour un fichier 
# ==============================
def predict_spatialisation_lstm(model, scaler, df, numeric_columns, seq_len=50):
    X_in = df[numeric_columns].fillna(0.0).values
    X_scaled =  scaler.transform(X_in)

    # features sans coords
    X_features = X_scaled[:, :-2]
    X_tensor = torch.tensor(X_features, dtype=torch.float32).unsqueeze(0)  # batch=1

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).squeeze(0).numpy()

    # remettre à l’échelle originale
    # on crée un vecteur complet pour inverse_transform
    padded = np.zeros((pred_scaled.shape[0], scaler.mean_.shape[0]))
    padded[:, :-2] = X_features
    padded[:, -2:] = pred_scaled
    pred = scaler.inverse_transform(padded)[:, -2:]

    return pred


if __name__ == "__main__":
    # ---- Choix des paramètres ----
    train_seqs = [1,2,3,4]   # séquences utilisées pour l’entraînement
    predict_seqs = [0]       # séquences pour lesquelles on prédit les coordonnées 
    data_dir = "resampled_results2"
    out_dir = "resampled_results2_predict_coord"
    os.makedirs(out_dir, exist_ok=True)

    # Récupération des fichiers d’entraînement
    DATA_TRAIN_PATHS = []
    for seq in train_seqs:
        pattern = os.path.join(data_dir, f"dataset_track*_seq{seq}.csv")
        for file_path in glob.glob(pattern):
            DATA_TRAIN_PATHS.append(file_path)

    # Récupération des fichiers pour les séquences à prédire
    DATA_PREDICT_PATHS = []
    for seq in predict_seqs:
        pattern = os.path.join(data_dir, f"dataset_track*_seq{seq}.csv")
        DATA_PREDICT_PATHS.extend(glob.glob(pattern))

    # Colonnes utilisées (les deux dernières sont x,y)
    numeric_columns = [
        "rms_Sample",
        "rms_Voc 1",
        "rms_Voc 2",
        "rms_Guitare",
        "rms_Basse",
        "rms_BatterieG",
        "rms_BatterieD",
        "beat",
        "x_Voc 1",
        "y_Voc 1"
    ]



    # ---- Entraînement ----
    model, scaler = run_training(
        DATA_TRAIN_PATHS,
        numeric_columns=numeric_columns,
        seq_len=50,
        batch_size=32,
        n_epochs=30   # plus tu montes, mieux c’est (si tu as le temps)
    )

    

    # ---- Prédiction pour les fichiers "_no_coord" ----
    instr_map = {"Sample" : 0, "Voc 1": 1, "Voc 2": 2, "Guitare": 3, "Basse" : 4, "BatterieG" : 5, "BatterieD" : 6}
    tracks_grouped = defaultdict(list)


    instr_name = "Voc 1"  # ici on prédit pour "Voc 1"
    instr_id = instr_map[instr_name]

    for file_path in DATA_PREDICT_PATHS:
        print(f"\nPrédiction pour : {file_path}")
        df_test = pd.read_csv(file_path)

        # Prédiction avec le LSTM
        coords_pred = predict_spatialisation_lstm(model, scaler, df_test, numeric_columns)

        # Remplissage des colonnes avec les coordonnées prédites
        df_test[f"x_{instr_name}"] = [c[0] for c in coords_pred]
        df_test[f"y_{instr_name}"] = [c[1] for c in coords_pred]

        # Nouveau chemin dans le dossier de sortie
        base_name = os.path.basename(file_path).replace(".csv", "_predicted_coord.csv")
        out_path = os.path.join(out_dir, base_name)

        df_test.to_csv(out_path, index=False)
        print(f"Coordonnées prédites enregistrées dans : {out_path}")

        # Ajout au regroupement par track
        m = re.search(r"track(\d+)_seq(\d+)", file_path)
        track_id = int(m.group(1)) if m else -1
        tracks_grouped[track_id].append(out_path)

    # ---- Conversion en données Max ----
    from convert_to_max_data import convert_to_max_data
    RESULT_FOLDER_MAX = "resampled_result_2_max_data"
    os.makedirs(RESULT_FOLDER_MAX, exist_ok=True)

    seq_file = os.path.join(RESULT_FOLDER_MAX, f"seq{instr_id + 1}.txt")
    if os.path.exists(seq_file):
        os.remove(seq_file)
    for track_id, csv_files in tracks_grouped.items():
        print(f"\nConversion Max pour track {track_id} avec {len(csv_files)} fichiers...")
        convert_to_max_data(csv_files=csv_files, track_id=track_id, instr_coord_id=instr_id)

from sklearn.preprocessing import LabelEncoder
def encode_datasets(file_paths, numeric_columns):
    """Charge les CSV, encode les colonnes non-numériques en entiers et renvoie les DataFrames encodés."""
    dfs = [pd.read_csv(fp) for fp in file_paths]
    encoders = {}

    # Construire les encoders globaux
    for col in numeric_columns:
        if any(not pd.api.types.is_numeric_dtype(df[col]) for df in dfs):
            unique_vals = pd.concat([df[col].astype(str) for df in dfs]).unique()
            le = LabelEncoder()
            le.fit(unique_vals)
            encoders[col] = le

    # Transformer chaque DataFrame
    encoded_dfs = []
    for df in dfs:
        for col, le in encoders.items():
            df[col] = le.transform(df[col].astype(str))
        encoded_dfs.append(df)

    return encoded_dfs


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def evaluate_model(model, scaler, data_paths, numeric_columns, coord_columns, seq_len=50):
    all_true, all_pred = [], []

    for path in data_paths:
        df = pd.read_csv(path)
        coords_true = df[coord_columns].values

        # prédiction
        coords_pred = predict_spatialisation_lstm(model, scaler, df, numeric_columns, seq_len=seq_len)

        all_true.append(coords_true[:len(coords_pred)])
        all_pred.append(coords_pred)

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    # métriques
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # distance euclidienne
    dist = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
    mean_dist = dist.mean()

    print(f"MSE   = {mse:.4f}")
    print(f"RMSE  = {rmse:.4f}")
    print(f"MAE   = {mae:.4f}")
    print(f"R²    = {r2:.4f}")
    print(f"Mean Euclidean Distance = {mean_dist:.4f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mean_dist": mean_dist}