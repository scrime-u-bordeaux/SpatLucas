import re
import os
import sys
import pandas as pd

indexInstrument = {
    0: "Sample",
    1: "Voc 1",
    2: "Voc 2",
    3: "Guitare",
    4: "Basse",
    5: "BatterieG",
    6: "BatterieD",
}

indexTitre = {
    1: "APOSTAT",
    2: "ECRAN DE FUMEE",
    3: "L'ENNEMI",
    4: "HYPNOSE",
    5: "COMMUNION",
    6: "FACE AUX GEANTS",
    7: "NOUVEAU DIABLE",
    8: "BALLADE ENTRE LES MINES",
    9: "TEMPS MORT",
}

# Fonction pour convertir un timecode en secondes
def timecode_to_seconds(tc):
    tc_str = str(tc)
    # Pattern timecode classique
    match = re.match(r"(\d+):(\d+):(\d+):(\d+)", tc_str)
    if match:
        h, m, s, f = map(int, match.groups())
        fps = 25
        return h * 3600 + m * 60 + s + f / fps
    # Pattern nombre à 3 décimales (ex: 12.345)
    match_decimal = re.match(r"(\d+)\.(\d{3})", tc_str)
    if match_decimal:
        sec = int(match_decimal.group(1))
        ms = int(match_decimal.group(2))
        return sec + ms / 1000
    return None

def get_regions_from_name(track_name_file):
    base_dir = os.path.dirname(__file__)  # le dossier contenant ce fichier
    csv_audio_path = os.path.join(base_dir, "Audio", f"{track_name_file}.csv")
    regions = []
    if os.path.exists(csv_audio_path):
        df_audio_csv = pd.read_csv(csv_audio_path)
        if not df_audio_csv.empty:
            # Ajout d'une première région fictive "Silence" avant le début de la musique
            first_start = timecode_to_seconds(df_audio_csv.iloc[0]["Start"])
            regions.append({"name": "Silence", "start": 0.0})

            for _, row in df_audio_csv.iterrows():
                start_sec = timecode_to_seconds(row["Start"])
                region = {"name": row["Name"], "start": start_sec}
                if start_sec is None or start_sec <= 0.0:
                    continue
                regions.append(region)
    else:
        print(f"Fichier CSV non trouvé : {csv_audio_path}")
    return sorted(regions, key=lambda x: x["start"])
        
def get_instrument_name(index):
    try:
        idx = int(index)
    except (ValueError, TypeError):
        return "Inconnu"
    return indexInstrument.get(idx, "Inconnu")


def get_track_name(index, reading_file=False):
    try:
        idx = int(index)
    except (ValueError, TypeError):
        return "Inconnu"
    name = indexTitre.get(idx, "Inconnu")
    if reading_file:
        name = name.lower().replace("'", "-").replace(" ", "-")
        name = name[0].upper() + name[1:] if name else name
    return name

def evaluate_range_dataset(csv):
    df = pd.read_csv(csv)
    for column in df.columns:
        try:
            min_val = df[column].min()
            max_val = df[column].max()
            print(f"{column}:")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
        except Exception as e:
            print(f"{column}: erreur lors de l’analyse ({e})")
    return df


