import pandas as pd
import itertools
import contextlib
import wave
from scipy.io import wavfile
import numpy as np
import os   
import re
import warnings
import sys


from Utils import get_track_name
from Utils import get_regions_from_name, indexInstrument
from Data_Visualization.audioRMS import detect_periods
import sys

"""
    Create Dataset from parsed data in order to train a model.
    We can specify the parameters to parse the data.
    For now we put only one instrument in the dataset (mostly the guitar).
"""

SEQ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'seq')
AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Audio')
RESAMPLE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'TestFolder')
RESAMPLE_DIR = os.path.join(os.path.dirname(__file__), "resampled_results")

TIME_SAMPLE = 0.02

# 0: "Voc 1"
# 1: "Voc 2"
# 2: "Guitare"
# 3: "Basse"
# 4: "BatterieG"
# 5: "BatterieD"

INSTRUMENT_IDX = 2

# 1: "APOSTAT"
# 2: "ECRAN DE FUMEE"
# 3: "L'ENNEMI"
# 4: "HYPNOSE"
# 5: "COMMUNION"
# 6: "FACE AUX GEANTS"
# 7: "NOUVEAU DIABLE"
# 8: "BALLADE ENTRE LES MINES"
# 9: "TEMPS MORT"

TRACK_IDX = 9

class Coord:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"{self.x, self.y}\n"

def get_audio_duration(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        return f.getnframes() / f.getframerate()


"""
    Resample input arrays (datas) according to the given times array,
    using a fixed time step (TIME_SAMPLE). All arrays must be of the same length.
    Parameters:
        times (list or array): Time points corresponding to the data.
        total_duration (float): Total duration to resample over.
        *datas (list or array): One or more arrays of data to resample, all of the same length as times.
    Returns: 
        tuple: (resampled_times, resampled_data)
            resampled_times: list of time points used for resampling
            resampled_data: list of values (or tuples if multiple datas) at each time step
    """
def resample(times, total_duration, *datas):
    if not datas:
        print("No data provided for resampling.")
        return [], []
    n = len(times)
    if any(len(d) != n for d in datas):
        raise ValueError("All input arrays must have the same length as times.")

    resampled_data = []
    resampled_times = []

    prev_time_window = TIME_SAMPLE
    idx = 0

    while prev_time_window <= total_duration:
        if idx < n and times[idx] <= prev_time_window:
            values = tuple(d[idx] for d in datas)
            idx += 1
        else:
            values = tuple(d[idx - 1] if idx > 0 else d[0] for d in datas)

        resampled_data.append(values[0] if len(values) == 1 else values)
        resampled_times.append(prev_time_window)
        prev_time_window += TIME_SAMPLE

    while prev_time_window <= total_duration:
        resampled_data.append(resampled_data[-1]) 
        resampled_times.append(prev_time_window)
        prev_time_window += TIME_SAMPLE

    return resampled_times, resampled_data


# SPAT DATA PARSING
"""
   Parse spatial data from a file and return the coordinates and speed.
   Parameters:
         print_times (bool): If True, print the times of the parsed data.
    Returns:
        tuple: (resampled_times, resampled_coord, resampled_speed)
            resampled_times: list of time points used for resampling
            resampled_coord: list of coordinates (x, y) at each time step
            resampled_speed: list of speed vectors (dx, dy) at each time step
   """
def parse_spat_data(print_times=False):
    filename = f"{indexInstrument[INSTRUMENT_IDX + 1]}.txt"
    df = pd.read_csv(os.path.join(SEQ_DIR, filename), header=None)

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

    track_index, track_name, figures = combined[TRACK_IDX - 1]
    audio_path = os.path.join(AUDIO_DIR, f"{track_name}.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier audio {track_name}.wav introuvable")
    total_duration = get_audio_duration(audio_path)

    real_times, x_coords, y_coords = zip(*[
        (float(parts[-3]) * total_duration, float(parts[-2]), float(parts[-1]))
        for value in figures if len(parts := value.split()) >= 3
    ])

    print(f"Nombre de points spatiaux : {len(real_times)}")


    resampled_times, resampled_coord = resample(real_times, total_duration, x_coords, y_coords)

    x_coords_np = np.array([pt[0] for pt in resampled_coord])
    y_coords_np = np.array([pt[1] for pt in resampled_coord])
    real_times_np = np.array(real_times)

    dx = np.diff(x_coords_np, prepend=x_coords_np[0])
    dy = np.diff(y_coords_np, prepend=y_coords_np[0])

    # dx = np.diff(x_coords_np)
    # dx = np.insert(dx, 0, 0.0)
    # dy = np.diff(y_coords_np)
    # dy = np.insert(dy, 0, 0.0)                                                      

    resampled_speed = list(zip(dx, dy))

    if not print_times:
        resampled_times = None

    return resampled_times, resampled_coord, resampled_speed

# AUDIO DATA PARSING

"""
    Compute the RMS values of the audio data.
    Parameters:
        data_instr (numpy array): The audio data for the instrument.
        rate (int): The sample rate of the audio data.
        window_sec (float): The size of the window in seconds for RMS calculation.
        alpha (float): The smoothing factor for the RMS calculation.
    Returns:
        tuple: (rms_values, window_sec)
            rms_values: numpy array of RMS values calculated over the windows
            window_sec: the size of the window in seconds used for RMS calculation
   """
def compute_rms(data_instr, rate, window_sec=TIME_SAMPLE, alpha=0.2):
    window_size = int(rate * window_sec)
    num_windows = int(len(data_instr) / window_size)
    rms_values = []
    times = []
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
        times.append(i * window_sec)
    
    return  np.array(times), np.array(rms_values)

"""
   parse audio data and return the RMS values and times.
   Parameters:
        detect_played_periods (bool): If True, detect periods where the instrument is played, else print raw RMS values.
    Returns:
        tuple: (times, rms_values)
            times: list of time points corresponding to the RMS values
            rms_values: list of RMS values at each time point
    """


def parse_audio_rms(detect_played_periods = False):
    track_name = get_track_name(TRACK_IDX, True)
    instrument_idx = INSTRUMENT_IDX + 1
    print(f"Analyse audio pour la piste : {track_name}")
    print(f"Analyse de l'instrument : {indexInstrument[instrument_idx]}")
    

   
    fname = os.path.join(AUDIO_DIR, f"{track_name}.wav")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, data = wavfile.read(fname)
    channel_idx = instrument_idx + 1  # Audio has a "clic" channel at index 0
    data_instr = data[:, channel_idx]



    times, rms_values = compute_rms(data_instr, rate)
    print("Nombres de valeurs RMS calculées : ", len(rms_values))


    # On souhaite remplir le tableau de valeur booleenne (instrument joué ou non joué)
    if(detect_played_periods):
        rms_values_norm = rms_values / np.max(rms_values) * 10 if np.max(rms_values) > 0 else rms_values

        joue_periods_filtrees, _, _ = detect_periods(rms_values_norm, TIME_SAMPLE)

        # The format is the following : (14.3, 77.98), (81.22, 192.46), (199.38, 222.0)
        # Write the times in an array and associate the first of the couple with the value True, and the second with the value False


        played_times = []
        played_values = []
        played_times.append(0.0)
        played_values.append(0)
        for start, end in joue_periods_filtrees:
            played_times.append(start)
            played_values.append(1)
            played_times.append(end)
            played_values.append(0)

        audio_path = os.path.join(AUDIO_DIR, f"{track_name}.wav")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio {track_name}.wav introuvable")
        total_duration = get_audio_duration(audio_path)
        resampled_times, resampled_played_values = resample(played_times, total_duration, played_values)
        time = resampled_times
        rms_values = resampled_played_values

    else:
        rms_values = rms_values
    return times, rms_values


"""
    Parse the regions from the audio file
    Returns:
        resampled_regions: list of regions with resampled times
   """
def parse_regions():
    track_name = get_track_name(TRACK_IDX, True)
    regions = get_regions_from_name(track_name)
    if(len(regions) == 0):
        print(f"No regions found for track {track_name}.")
        return []

    region_names_raw = [region["name"] for region in regions]
    
    # Extraction entre apostrophes + remplacement des espaces par des underscores
    region_names = [
        re.search(r"'(.*?)'", name).group(1).replace(' ', '_')
        if re.search(r"'(.*?)'", name) else name.replace(' ', '_')
        for name in region_names_raw
    ]
    
    region_starts = [region["start"] for region in regions]
    
    audio_path = os.path.join(AUDIO_DIR, f"{track_name}.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier audio {track_name}.wav introuvable")
    
    total_duration = get_audio_duration(audio_path)

    # Resample avec la nouvelle version de la fonction
    _, resampled_regions = resample(region_starts, total_duration, region_names)
    return resampled_regions

"""
    Parse the BPM and calculate the positions of beats and measures.
    Parameters:
        beats_per_measure (int): Number of beats per measure, default is 4.
    Returns:
        tuple: (resampled_beats, resampled_measures)
            resampled_beats: list of resampled beats with their times and units
            resampled_measures: list of resampled measures with their times and units
    """
def parse_bpm_and_calculate_positions(beats_per_measure=4):
    track_name_file = get_track_name(TRACK_IDX, True)
    track_name = get_track_name(TRACK_IDX)
    
    audio_path = os.path.join(AUDIO_DIR, f"{track_name_file}.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier audio {track_name_file}.wav introuvable")

    total_duration = get_audio_duration(audio_path)

    df = pd.read_csv("BPM_tracks.csv")
    track_row = df[df["name"] == track_name]
    if track_row.empty:
        raise ValueError(f"Track '{track_name}' non trouvé dans BPM_tracks.csv")

    bpm = float(track_row["bpm"].values[0])
    beat_duration = 60 / bpm
    total_beats = int(total_duration / beat_duration)

    # Génère les timecodes de chaque beat
    beat_timecodes = [round(i * beat_duration, 6) for i in range(total_beats)]

    beat_times = []
    beat_units = []
    measure_times = []
    measure_units = []

    for i, t in enumerate(beat_timecodes):
        beat_in_measure = i % beats_per_measure + 1
        measure_number = i // beats_per_measure + 1

        beat_times.append(t)
        beat_units.append(beat_in_measure)

        if beat_in_measure == 1:
            measure_times.append(t)
            measure_units.append(measure_number)

    # Resample beats
    _, resampled_beats = resample(beat_times, total_duration, beat_units)

    # Resample measures
    _, resampled_measures = resample(measure_times, total_duration, measure_units)

    return resampled_beats, resampled_measures

"""   
    Create a CSV file with all values from the parsed data.
    Parameters:
        all_tracks (bool): If True, create a CSV for all tracks, else for a single track.
    Returns:
        None
"""
def create_csv_all_values(all_tracks=True):
    if(all_tracks):
        tracks_idx = list(range(1,10))
    else:
        track_idx = TRACK_IDX
        tracks_idx = []
        tracks_idx.append(track_idx)

    for track_idx in tracks_idx:
        times, audio_data = parse_audio_rms(detect_played_periods=True)
        _, spat_coord, spat_speed = parse_spat_data() # First return is times, we can use it later if needed

        regions_data = parse_regions()
        beats, measures = parse_bpm_and_calculate_positions()

        if(len(times) == len(audio_data)  == len(spat_coord) == len(spat_speed) 
           == len(regions_data) == len(measures)) == len(beats): print("Même longueur de fichier")


        if len(audio_data) == len(spat_coord) == len(regions_data):
            print("Création du csv...")
        else:
            raise ValueError("Les longueurs des fichiers parsés ne correspondent pas.")

        track_name_file = get_track_name(track_idx, True) 
        output_path = os.path.join(RESAMPLE_DIR, f"dataset_all_value_{track_name_file}.csv")
        
        with open(output_path, "w") as out_file:
            # Header
            out_file.write("time,rms,region,temps,mesures,x,y\n")
            for (time, rms, (x, y), region, beat, measure) in zip(times, audio_data, spat_coord, regions_data, beats, measures):
                out_file.write(f"{time:.2f},{rms},{region},{beat},{measure},{x},{y}\n")

        print(f"Fichier CSV unifié créé dans : {output_path}")
    
if __name__ == "__main__":
    all_tracks = False
    create_csv_all_values(all_tracks)