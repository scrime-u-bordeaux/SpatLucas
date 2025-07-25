import os
import itertools
import pandas as pd
import wave
import contextlib
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Utils import get_instrument_name, get_regions_from_name

SEQ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'seq')
AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Audio')
RESAMPLE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'newSeq')

def resample_figures(figures, total_duration):
    result = [
        [float(parts[-3]), float(parts[-2]), float(parts[-1])]
        for value in figures
        if len(parts := value.split()) >= 3
    ]
    resampled = []
    prev_time = 0.0
    resampled.append([0, 0])
    for rel_time, x, y in result:
        if rel_time <= 0 or rel_time >= total_duration:
            continue
        time = rel_time * total_duration
        t = prev_time + 0.01
        while t < time:
            resampled.append([x, y])
            t += 0.01
        prev_time = time
        resampled.append([x, y])
    return result, resampled

def process_track(
    instrument_idx=None,
    selected_num_track=None,
    seq_dir=None,
    audio_dir=None,
    resample_output_dir=None,
    all_instruments=False
):
    if seq_dir is None:
        seq_dir = SEQ_DIR
    if audio_dir is None:
        audio_dir = AUDIO_DIR
    if resample_output_dir is None:
        resample_output_dir = RESAMPLE_OUTPUT_DIR

    data = {
        filename: pd.read_csv(os.path.join(seq_dir, filename), header=None)
        for filename in os.listdir(seq_dir)
        if filename.endswith('.txt')
    }

    ## =========== Tous les instruments ============ ##

    if all_instruments:
        instrument_files = sorted(data.keys())
        combined_raw = []
        combined_resampled = []
        total_durations = {}
        for instrument_file in instrument_files:
            df = data[instrument_file]
            track_indices, track_names, figures_groups = [], [], []
            current_figures = []

            for value in df.iloc[:, 1].values:
                value_str = str(value)
                if value_str.strip().startswith("id"):
                    if current_figures:
                        figures_groups.append(current_figures)
                        current_figures = []
                    track_part = value_str.split('-', 1)[0].replace('id', '').strip()
                    track_index = int(track_part) if track_part.isdigit() else None
                    track_indices.append(track_index)
                    track_name = value_str.split('-', 1)[-1].strip().rstrip(';')
                    track_names.append(track_name)
                else:
                    current_figures.append(' '.join(value_str.rstrip(';').strip().split()))
            if current_figures:
                figures_groups.append(current_figures)

            combined = sorted(
                itertools.zip_longest(track_indices, track_names, figures_groups),
                key=lambda x: (x[0] if x[0] is not None else float('inf'))
            )

            if selected_num_track is None:
                raise ValueError("selected_num_track doit être spécifié")
            selected_track_index = int(selected_num_track) - 1
            if selected_track_index >= len(combined):
                continue

            track_index, track_name, figures = combined[selected_track_index]

            audio_path = os.path.join(audio_dir, f"{track_name}.wav")
            if not os.path.exists(audio_path):
                continue

            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                total_duration = f.getnframes() / f.getframerate()
            total_durations[track_name] = total_duration

            raw_result, resampled = resample_figures(figures, total_duration)
            combined_raw.extend(raw_result)
            combined_resampled.extend(resampled)

        print(len(combined_raw), "figures combinées pour", selected_num_track, "piste(s)")

        return {
            "track_name": track_name,
            "instrument_name": "ALL",
            "result": combined_raw,
            "resampled_result": combined_resampled,
            "total_duration": total_duration,  # multiple tracks, so undefined
            "regions": get_regions_from_name(track_name)
        }

    ## =========== Un seul instrument ============ ##

    else:
        if instrument_idx is None:
            raise ValueError("instrument_idx doit être spécifié si all_instruments=False")

        instrument_name = get_instrument_name(instrument_idx)
        instrument_file = f"{instrument_name}.txt"

        if instrument_file not in data:
            raise FileNotFoundError(f"{instrument_file} n'existe pas dans le dossier.")

        df = data[instrument_file]
        track_indices, track_names, figures_groups = [], [], []
        current_figures = []

        for value in df.iloc[:, 1].values:
            value_str = str(value)
            if value_str.strip().startswith("id"):
                if current_figures:
                    figures_groups.append(current_figures)
                    current_figures = []
                track_part = value_str.split('-', 1)[0].replace('id', '').strip()
                track_index = int(track_part) if track_part.isdigit() else None
                track_indices.append(track_index)
                track_name = value_str.split('-', 1)[-1].strip().rstrip(';')
                track_names.append(track_name)
            else:
                current_figures.append(' '.join(value_str.rstrip(';').strip().split()))
        if current_figures:
            figures_groups.append(current_figures)

        combined = sorted(
            itertools.zip_longest(track_indices, track_names, figures_groups),
            key=lambda x: (x[0] if x[0] is not None else float('inf'))
        )

        
        if selected_num_track is None:
            raise ValueError("selected_num_track doit être spécifié")
        selected_track_index = int(selected_num_track) - 1
        if selected_track_index >= len(combined):
            raise IndexError("Numéro de piste sélectionné hors limites")

        track_index, track_name, figures = combined[selected_track_index]
        audio_path = os.path.join(audio_dir, f"{track_name}.wav")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio {track_name}.wav introuvable")

        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            total_duration = f.getnframes() / f.getframerate()

        raw_result, resampled = resample_figures(figures, total_duration)

        # Écriture fichier resamplé (si différent ou inexistant)
        resample_output_path = os.path.join(resample_output_dir, f"{instrument_name} - {track_name}.txt")
        os.makedirs(resample_output_dir, exist_ok=True)
        lines = [f"{x} {y}\n" for x, y in resampled]

        write_file = True
        if os.path.exists(resample_output_path):
            with open(resample_output_path, 'r', encoding='utf-8') as f:
                if f.readlines() == lines:
                    write_file = False

        if write_file:
            with open(resample_output_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"File created/replaced: {resample_output_path}")

        return {
            "track_name": track_name,
            "instrument_name": instrument_name,
            "result": raw_result,
            "resampled_result": resampled,
            "total_duration": total_duration,
            "regions": get_regions_from_name(track_name)
        }