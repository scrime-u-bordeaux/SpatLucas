import os
import numpy as np
import pandas as pd
import warnings
import re
import contextlib, wave
import sys
from scipy.io import wavfile

num_dataset = 0
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
from Utils import get_regions_from_name
from Utils import get_track_name
from Utils import get_instrument_name

"""
Fournit la durée audio du fichier "path"
"""
def get_audio_duration(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        return f.getnframes() / f.getframerate()

"""
Calcule les valeurs RMS pour les données audio.
Paramètres:
------------
    data_instr: Les données audio pour l'instrument.
    rate: La fréquence d'échantillonnage des données audio.
    window_sec: La durée de la fenêtre en secondes pour le calcul du RMS.
    alpha: Le facteur de lissage pour le calcul du RMS (par défaut 0.2).
"""
def compute_rms(data_instr, rate, window_sec, alpha=0.2):
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
    return np.array(rms_values)

"""
Rééchantillonne les données à intervalles réguliers.
Paramètres:
------------
    times: Les temps des données originales.
    total_duration: La durée totale de l'audio.
    *datas: Les données à rééchantillonner.
    step: L'intervalle de temps pour le rééchantillonnage (par défaut 0.01s).
"""
def resample(times, total_duration, *datas, step=0.01):
    resampled_data, resampled_times = [], []
    prev_time_window, idx, n = step, 0, len(times)
    while prev_time_window <= total_duration:
        if idx < n and times[idx] <= prev_time_window:
            values = tuple(d[idx] for d in datas)
            idx += 1
        else:
            values = tuple(d[idx - 1] if idx > 0 else d[0] for d in datas)
        resampled_data.append(values[0] if len(values) == 1 else values)
        resampled_times.append(prev_time_window)
        prev_time_window += step
    return np.array(resampled_times), np.array(resampled_data)

"""
Construit un dataset pour un instrument cible donné, en utilisant les indices RMS spécifiés.
Paramètres:
------------
    track_id: L'ID de la piste audio.
    rms_indices: Liste des indices des instruments pour lesquels calculer le RMS.
    target_idx: L'indice de l'instrument cible pour lequel prédire les coordonnées.
    start_time: Le temps de début pour le dataset (par défaut 1s).
    end_time: Le temps de fin pour le dataset (par défaut toute la durée).
    fs: La fréquence d'échantillonnage pour le dataset (par défaut 10Hz).
    seq_dir: Le répertoire contenant les fichiers de séquences spatiales (par défaut "seq").
    audio_dir: Le répertoire contenant les fichiers audio (par défaut "Audio").
"""
def build_dataset(track_id, rms_indices, target_idx,
                  start_time=1, end_time=None, fs=10,
                  seq_dir="seq", audio_dir="Audio", bpm_file="BPM_tracks.csv",
                  out_dir="Data_Training/sequences_datasets",
                  include_time=True, include_beats=True, include_measures=True, 
                  include_regions=True):
    
    global num_dataset
    # --- Charger le fichier spat ---
    txt_path = os.path.join(seq_dir, get_instrument_name(target_idx) + ".txt")
    df = pd.read_csv(txt_path, header=None)

    # Trouver le groupe correspondant
    figures, current_figures, current_track, track_name = [], [], None, None
    for value in df.iloc[:, 1].values:
        value_str = str(value).strip()
        if value_str.startswith("id"):
            if current_track == track_id and current_figures:
                figures = current_figures
                break
            current_track = int(value_str.split('-', 1)[0].replace("id", "").strip())
            current_track_name = value_str.split('-', 1)[-1].strip().rstrip(';')
            current_figures = []
        else:
            current_figures.append(' '.join(value_str.rstrip(';').split()))
    if not figures and current_track == track_id:
        figures = current_figures
        track_name = current_track_name
    elif figures:
        track_name = current_track_name

    audio_path = os.path.join(audio_dir, f"{track_name}.wav")
    total_duration = get_audio_duration(audio_path)
    if end_time is None or end_time > total_duration:
        end_time = total_duration

    # --- Extraire coords ---
    times, xs, ys = [], [], []
    for value in figures:
        parts = value.split()
        if len(parts) >= 3:
            t = float(parts[-3]) * total_duration
            times.append(t)
            xs.append(float(parts[-2]))
            ys.append(float(parts[-1]))
    times, xs, ys = np.array(times), np.array(xs), np.array(ys)

    # Grille régulière
    grid = np.arange(start_time, end_time, 1/fs)

    # --- RMS ---
    rms_dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, data = wavfile.read(audio_path)
    for idx in rms_indices:
        data_instr = data[:, idx] if data.ndim > 1 else data
        rms_values = compute_rms(data_instr, rate, window_sec=1/fs)
        rms_times = np.arange(len(rms_values)) * (1/fs)
        rms_interp = np.interp(grid, rms_times, rms_values)
        rms_dict[f"rms_{get_instrument_name(idx)}"] = rms_interp

    # --- Position cible ---
    x_interp = np.interp(grid, times, xs)
    y_interp = np.interp(grid, times, ys)

    # --- Construction DataFrame ---
    df_out = pd.DataFrame(rms_dict)

    if include_time:
        df_out["time_sec"] = grid

    if include_beats or include_measures:
        df_bpm = pd.read_csv(bpm_file)
        track_row = df_bpm[df_bpm["name"] == get_track_name(track_id)]
        bpm = float(track_row["bpm"].values[0])
        beat_dur = 60 / bpm
        total_beats = int(total_duration / beat_dur)
        beat_times = [i * beat_dur for i in range(total_beats)]
        beat_units = [(i % 4) + 1 for i in range(total_beats)]
        measure_times = [i * beat_dur for i in range(0, total_beats, 4)]
        measure_units = [i//4 + 1 for i in range(total_beats) if i % 4 == 0]
        _, beats = resample(beat_times, total_duration, beat_units, step=1/fs)
        _, measures = resample(measure_times, total_duration, measure_units, step=1/fs)
        beats, measures = beats[:len(grid)], measures[:len(grid)]
        if include_beats:
            df_out["beat"] = beats
        if include_measures:
            df_out["measure"] = measures

    if include_regions:
        regions_data = get_regions_from_name(track_name)
        region_names_raw = [r["name"] for r in regions_data]
        region_names = [
            re.search(r"'(.*?)'", name).group(1).replace(' ', '_')
            if re.search(r"'(.*?)'", name) else name.replace(' ', '_')
            for name in region_names_raw
        ]
        region_starts = np.array([float(r["start"]) for r in regions_data])
        idxs = np.searchsorted(region_starts, grid, side="right") - 1
        regions_resampled = np.array([
            region_names[i] if i >= 0 else ""
            for i in idxs
        ])
        df_out["region"] = regions_resampled[:len(grid)]

    # --- Ajouter coords réelles ---
    target_name = get_instrument_name(target_idx)
    df_out[f"x_{target_name}"] = x_interp
    df_out[f"y_{target_name}"] = y_interp

    # --- Réordonner les colonnes ---
    ordered_cols = []
    if include_time:
        ordered_cols.append("time_sec")
    ordered_cols.extend([c for c in df_out.columns if c.startswith("rms_")])
    if include_regions:
        ordered_cols.append("region")
    if include_beats:
        ordered_cols.append("beat")
    if include_measures:
        ordered_cols.append("measure")
    ordered_cols.extend([f"x_{target_name}", f"y_{target_name}"])

    df_out = df_out[ordered_cols]
    
    out_path = os.path.join(out_dir, f"dataset_track{track_id}_seq{num_dataset}.csv")
    df_out.round(3).to_csv(out_path, index=False)
    print(f"Dataset créé : {out_path}")

    num_dataset += 1
    return df_out

import glob 
"""
Supprime tous les fichiers datasets générés dans le dossier spécifié.
Paramètres:
------------
    out_dir: Le répertoire contenant les fichiers datasets à supprimer (par défaut "Data_Training/sequences_datasets").
"""
def clear_datasets(out_dir="Data_Training/sequences_datasets"):
    """Supprime tous les fichiers datasets générés dans le dossier spécifié."""
    files = glob.glob(os.path.join(out_dir, "dataset_track*_seq*.csv"))
    for f in files:
        os.remove(f)
    print(f"{len(files)} fichiers supprimés de {out_dir}")



if __name__ == "__main__":
    # Supprime les anciens datasets
    out_dir = "Data_Training/sequences_datasets"
    clear_datasets(out_dir=out_dir)
    # Ne pas préciser start_time et end_time = toute la durée

    ## Nouveau diable
    build_dataset(track_id=7, rms_indices=[0,1,2,3,4,5,6], target_idx=1, start_time=230, end_time=245, fs=10)
    build_dataset(track_id=7, rms_indices=[0,1,2,3,4,5,6], target_idx=1, start_time=109, end_time=119, fs=10)
    build_dataset(track_id=7, rms_indices=[0,1,2,3,4,5,6], target_idx=1, start_time=77, end_time=88, fs=10)
    build_dataset(track_id=7, rms_indices=[0,1,2,3,4,5,6], target_idx=1, start_time=129, end_time=141, fs=10)
    ## Hypnose
    # build_dataset(track_id=4, rms_indices=[0,1,2,3,4,5,6], target_idx=1, start_time=193, end_time=205, fs=10, add_without_coord=True)
    # for i in range(1, 9):
    #     # build_dataset(track_id=i, rms_indices=[0,1,2,3,4,5,6], target_idx=1, fs=10, add_without_coord=True, include_measures=True)
    #     build_dataset(track_id=i, rms_indices=[0,1,2,3,4,5,6], target_idx=1, fs=10)
    build_dataset(track_id=7, rms_indices=[0,1,2,3,4,5,6], target_idx=1, fs=10)
    



    