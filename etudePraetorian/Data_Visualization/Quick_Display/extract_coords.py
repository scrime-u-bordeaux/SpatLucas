import os
import wave
import contextlib
import numpy as np
import pandas as pd

# Remonter de deux dossiers pour que les chemins relatifs fonctionnent
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- Variables à adapter ---
TXT_FOLDER   = "seq/"   # fichier texte contenant les données spatiales
AUDIO_DIR = "Audio/"
TRACK_IDX = 7
INSTR_IDX = 1          # audio associé (pour avoir la durée totale)
FS         = 10                   # fréquence d’échantillonnage (points par seconde)
START_TIME = 109                        # début en secondes
END_TIME   = 118                        # fin en secondes (None = fin du morceau)

def extract_coords(track_id=TRACK_IDX, instr_idx=INSTR_IDX, fs=FS, start_time=START_TIME, end_time=END_TIME):
    """
    Extrait la séquence de coordonnées et vitesses d'un morceau choisi.

    Parameters
    ----------
    track_id : int, optionnel
        L'identifiant du morceau (ex : 3 si la ligne commence par "id 3-...").
    instr_idx : int, optionnel
        L'indice de l'instrument.
    fs : int, optionnel
        Fréquence d’échantillonnage (points par seconde).
    start_time : float, optionnel
        Début en secondes.
    end_time : float ou None, optionnel
        Fin en secondes (None = fin du morceau).
    """
    # --- Lecture du fichier texte ---
    indexInstrument = {
        0: "Sample",
        1: "Voc 1",
        2: "Voc 2",
        3: "Guitare",
        4: "Basse",
        5: "BatterieG",
        6: "BatterieD",
    }


    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, "../../.."))  # 3 niveaux au-dessus
    instrument_name = indexInstrument.get(instr_idx, f"Instr{instr_idx}")
    txt_path = os.path.join(parent_dir, TXT_FOLDER, instrument_name + ".txt")
    df = pd.read_csv(txt_path, header=None)
    # Séparer en groupes par "id ..."
    track_indices, track_names, figures_groups = [], [], []
    current_figures = []

    for value in df.iloc[:, 1].values:
        value_str = str(value)
        if value_str.strip().startswith("id"):
            if current_figures:
                figures_groups.append(current_figures)
                current_figures = []
            # Exemple : "id 3-Apostat;"
            track_part = value_str.split('-', 1)[0].replace('id', '').strip()
            idx = int(track_part) if track_part.isdigit() else None
            name = value_str.split('-', 1)[-1].strip().rstrip(';')
            track_indices.append(idx)
            track_names.append(name)
        else:
            current_figures.append(' '.join(value_str.rstrip(';').strip().split()))
    if current_figures:
        figures_groups.append(current_figures)

    # Chercher le morceau demandé
    selected_idx = None
    for i, (idx, name) in enumerate(zip(track_indices, track_names)):
        if track_id is not None and idx == track_id:
            selected_idx = i
            break

    if selected_idx is None:
        raise ValueError("Morceau non trouvé dans le fichier.")

    figures = figures_groups[selected_idx]
    chosen_name = track_names[selected_idx]
    chosen_idx = track_indices[selected_idx]

    # Afficher l'information sur le morceau et l'instrument
    print(f"Analyse du track : id={chosen_idx}, nom='{chosen_name}' (instrument : '{os.path.splitext(os.path.basename(instrument_name))[0]}')")

    # Charger audio associé
    audio_path = os.path.join(parent_dir, AUDIO_DIR, f"{chosen_name}.wav")
    audio_path = os.path.normpath(audio_path)  # normalise les slashes
    print(parent_dir)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")
    total_duration = get_audio_duration(audio_path)

    if end_time is None or end_time > total_duration:
        end_time = total_duration

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


    # --- Création de la nouvelle grille de temps ---
    new_times = np.arange(start_time, end_time, 1.0 / fs)

    # --- Interpolation des coordonnées ---
    x_interp = np.interp(new_times, times, xs)
    y_interp = np.interp(new_times, times, ys)
    coords = np.stack([x_interp, y_interp], axis=1)
    coords = np.round(coords, 4)

    # --- Calcul des vitesses (différences discrètes) ---
    dx = np.diff(x_interp, prepend=x_interp[0])
    dy = np.diff(y_interp, prepend=y_interp[0])
    speeds = np.stack([dx, dy], axis=1)
    print(f"Nombre de coordonnées extraites : {len(coords)}")
    print(f"Plage de temps : {new_times[0]:.2f}s à {new_times[-1]:.2f}s, fréquence d'échantillonnage : {fs} Hz")
    return {
        "name": chosen_name,
        "times": new_times,
        "coords": coords,
        "speeds": speeds
    }

def get_audio_duration(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        total_duration = f.getnframes() / f.getframerate()
    return total_duration


import matplotlib.pyplot as plt

def plot_coords(times, coords, title="Trajectoire spatiale"):
    """
    Affiche la trajectoire spatiale sur un graphique.

    Parameters
    ----------
    times : array-like
        Les temps associés aux coordonnées.
    coords : array-like
        Tableau Nx2 des coordonnées (x, y).
    title : str
        Titre du graphique.
    """
    xs, ys = coords[:, 0], coords[:, 1]
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, marker='o', linestyle='-', label='Trajectoire')
    plt.scatter(xs[0], ys[0], color='green', label='Début')
    plt.scatter(xs[-1], ys[-1], color='red', label='Fin')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_speed(times, speeds, title="Vitesse en fonction du temps"):
    """
    Affiche la norme de la vitesse en fonction du temps.

    Parameters
    ----------
    times : array-like
        Les temps associés aux vitesses.
    speeds : array-like
        Tableau Nx2 des vitesses (dx, dy).
    title : str
        Titre du graphique.
    """
    speed_norm = np.linalg.norm(speeds, axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(times, speed_norm, label='Vitesse')
    plt.xlabel('Temps (s)')
    plt.ylabel('Vitesse (unité/s)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    seq = extract_coords()
    print("Séquence extraite :", seq["coords"].shape)
    # Afficher le temps à côté des coordonnées
    # for t, (x, y) in zip(seq["times"], seq["coords"]):
    #     print(f"t={t:.3f}s : x={x}, y={y}")
    plot_coords(seq["times"], seq['coords'])
    plot_speed(seq['times'], seq["speeds"])


    