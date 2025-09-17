import numpy as np
import warnings
from scipy.io import wavfile
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

def compute_rms(data_instr, rate, window_sec=0.04, alpha=0.2):
    window_size = int(rate * window_sec)
    num_windows = int(len(data_instr) / window_size)
    rms_values = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        segment = data_instr[start:end]
        if len(segment) == 0:
            continue
        rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2))
        if rms_values:
            rms = alpha * rms + (1 - alpha) * rms_values[-1]
        rms_values.append(rms)
    return np.array(rms_values), window_sec

if __name__ == "__main__":
    print(parent_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio_path = os.path.join(parent_dir, "Audio", "Nouveau-diable.wav")
        rate, data = wavfile.read(audio_path)
    channel_idx =3 
    data_instr = data[:, channel_idx]
    rms_values, window_sec = compute_rms(data_instr, rate)
    
    import matplotlib.pyplot as plt

    times = np.arange(len(rms_values)) * window_sec
    mask = (times >= 105) & (times <= 120)
    plt.figure(figsize=(10, 4))
    plt.plot(times[mask], rms_values[mask])
    plt.xlabel("Time (s)")
    plt.ylabel("RMS")
    plt.title("RMS over Time (from 105s to 120s)")
    plt.tight_layout()
    plt.show()

import os
import wave
import contextlib
import numpy as np
import pandas as pd

# --- Variables à adapter ---
TXT_PATH   = "chemin/vers/fichier.txt"   # fichier texte contenant les données spatiales
AUDIO_PATH = "chemin/vers/fichier.wav"   # audio associé (pour avoir la durée totale)
FS         = 20                          # fréquence d’échantillonnage (points par seconde)
START_TIME = 5.0                         # début en secondes
END_TIME   = 15.0                        # fin en secondes (None = fin du morceau)

def extract_coords():
    # --- Obtenir la durée du fichier audio ---
    with contextlib.closing(wave.open(AUDIO_PATH, 'r')) as f:
        total_duration = f.getnframes() / f.getframerate()

    if END_TIME is None or END_TIME > total_duration:
        END_TIME = total_duration

    # --- Lecture du fichier texte ---
    df = pd.read_csv(TXT_PATH, header=None)

    # Récupérer toutes les "figures" (positions)
    figures = []
    current_figures = []

    for value in df.iloc[:, 1].values:
        value_str = str(value)
        if value_str.strip().startswith("id"):
            if current_figures:
                figures.extend(current_figures)
                current_figures = []
        else:
            current_figures.append(' '.join(value_str.rstrip(';').strip().split()))

    if current_figures:
        figures.extend(current_figures)

    # --- Extraction des temps normalisés et coords ---
    times, xs, ys = [], [], []
    for value in figures:
        parts = value.split()
        if len(parts) >= 3:
            t_norm = float(parts[-3])  # facteur entre 0 et 1
            x = float(parts[-2])
            y = float(parts[-1])
            times.append(t_norm * total_duration)  # convertir en secondes
            xs.append(x)
            ys.append(y)

    times = np.array(times)
    xs = np.array(xs)
    ys = np.array(ys)

    print(f"Nombre de points spatiaux lus : {len(times)}")

    # --- Création de la nouvelle grille de temps ---
    new_times = np.arange(START_TIME, END_TIME, 1.0 / FS)

    # --- Interpolation des coordonnées ---
    x_interp = np.interp(new_times, times, xs)
    y_interp = np.interp(new_times, times, ys)
    coords = np.stack([x_interp, y_interp], axis=1)

    # --- Calcul des vitesses (différences discrètes) ---
    dx = np.diff(x_interp, prepend=x_interp[0])
    dy = np.diff(y_interp, prepend=y_interp[0])
    speeds = np.stack([dx, dy], axis=1)

    # --- Résultat ---
    print("Séquence extraite :")
    print("Temps :", new_times.shape)
    print("Coords :", coords.shape)
    print("Vitesses :", speeds.shape)