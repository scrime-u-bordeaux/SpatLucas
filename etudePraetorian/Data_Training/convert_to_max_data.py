import os
import pandas as pd
from pathlib import Path
import soundfile as sf

track_id_map = {
    1: "Apostat",
    2: "Ecran-de-fumee",
    3: "L-ennemi",
    4: "Hypnose",
    5: "Communion",
    6: "Face-aux-geants",
    7: "Nouveau-diable",
    8: "Ballade-entre-les-mines",
    9: "Temps-mort"
}

def convert_to_max_data(csv_files, track_id, instr_coord_id):
    """
    Convertit une liste de CSV en format Max/MSP et ajoute les données
    dans le fichier seq<instr>.txt correspondant.
    L'index à gauche continue de s'incrémenter (pas de reset).
    """

    RESULT_FOLDER_MAX = "max_data_predicted"
    os.makedirs(RESULT_FOLDER_MAX, exist_ok=True)

    # Fichier de sortie (un seul par instrument)
    seq_file = os.path.join(RESULT_FOLDER_MAX, f"seq{instr_coord_id + 1}.txt")

    # Chercher le dernier index déjà présent dans le fichier
    last_index = 0
    if os.path.exists(seq_file):
        with open(seq_file, "r") as f:
            lines = f.readlines()
            if lines:
                try:
                    last_index = int(lines[-1].split(",")[0])
                except Exception:
                    last_index = 0

    # Fusionner tous les CSV
    all_data = [pd.read_csv(f) for f in csv_files]
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values(by="time_sec").reset_index(drop=True)

    # Chemin audio
    audio_dir = Path.cwd().parent / "Audio"
    audio_file = audio_dir / f"{track_id_map[track_id]}.wav"
    if not audio_file.exists():
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_file}")

    # Durée du morceau
    with sf.SoundFile(str(audio_file)) as f:
        duration = len(f) / f.samplerate

    # Colonnes x et y
    x_col = [c for c in df_all.columns if c.startswith("x_")][0]
    y_col = [c for c in df_all.columns if c.startswith("y_")][0]

    output_lines = []
    index = last_index + 1

    # Ligne d'ID du morceau
    output_lines.append(f"{index}, id 0{track_id}-{track_id_map[track_id]};")
    index += 1

    last_x, last_y = df_all.iloc[0][x_col], df_all.iloc[0][y_col]
    norm_time = df_all.iloc[0]["time_sec"] / duration
    output_lines.append(f"{index}, {norm_time:.6f} {last_x:.6f} {last_y:.6f};")
    index += 1

    for i in range(1, len(df_all)):
        x, y = df_all.iloc[i][x_col], df_all.iloc[i][y_col]
        norm_time = df_all.iloc[i]["time_sec"] / duration
        if x != last_x or y != last_y:
            output_lines.append(f"{index}, {norm_time:.6f} {x:.6f} {y:.6f};")
            last_x, last_y = x, y
            index += 1

    # Ajouter les nouvelles lignes à la fin du fichier
    with open(seq_file, "a") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"Ajouté {len(output_lines)} lignes dans {seq_file} (dernier index = {index - 1})")
