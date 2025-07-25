import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from getData import process_track
from scipy.io import wavfile
import warnings

"""
Compute the RMS values for the audio data.
Parameters:
    data_instr: The audio data for the instrument.
    rate: The sample rate of the audio data.
    window_sec: The size of the window in seconds (default is 0.01).
    alpha: The smoothing factor for the RMS calculation (default is 0.2).
Returns:
    The computed RMS values and the size of the window in seconds."""
def compute_rms(data_instr, rate, window_sec=0.01, alpha=0.2):
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

"""
Normalize the RMS values to a scale of 0 to 10.
Parameters:
    rms_values: The RMS values to normalize.
    scale: The scale to normalize to (default is 10).
Returns:
    The normalized RMS values.
    """
def normalize_rms(rms_values, scale=10):
    rms_max = np.max(rms_values)
    if rms_max > 0:
        return rms_values / rms_max * scale
    else:
        return rms_values

"""
Detect periods of silence and play for the instrument based on RMS values.
Parameters:
    rms_values_norm: The normalized RMS values.
    window_sec: The size of the window in seconds.
    rms_threshold: Threshold for RMS to consider as played.
    min_silence_duration: Minimum duration of silence to consider as not played.
    min_play_duration: Minimum duration of play to consider as played.
Returns:
    filtered_play_periods: Filtered periods where the instrument is played.
    short_play_periods: Periods where the instrument is played for a short duration.
    silence_periods: Periods where the instrument is not played.
"""
def detect_periods(rms_values_norm, window_sec, rms_threshold=1, min_silence_duration=2, min_play_duration=2):
    is_playing = rms_values_norm > rms_threshold
    silence_periods = []
    start_idx = None
    for idx, val in enumerate(is_playing):
        if not val and start_idx is None:
            start_idx = idx
        elif val and start_idx is not None:
            duration = (idx - start_idx) * window_sec
            if duration >= min_silence_duration:
                silence_periods.append((start_idx * window_sec, idx * window_sec))
            start_idx = None
    if start_idx is not None:
        duration = (len(is_playing) - start_idx) * window_sec
        if duration >= min_silence_duration:
            silence_periods.append((start_idx * window_sec, len(is_playing) * window_sec))
    play_periods = []
    prev_end = 0.0
    for start, end in silence_periods:
        if start > prev_end:
            play_periods.append((prev_end, start))
        prev_end = end
    if prev_end < len(is_playing) * window_sec:
        play_periods.append((prev_end, len(is_playing) * window_sec))
    filtered_play_periods = []
    short_play_periods = []
    for start, end in play_periods:
        if (end - start) >= min_play_duration:
            filtered_play_periods.append((start, end))
        else:
            short_play_periods.append((start, end))
    return filtered_play_periods, short_play_periods, silence_periods

"""
Plot the RMS values and save the figure.
Parameters:
    rms_values_norm: The normalized RMS values.
    window_sec: The size of the window in seconds.
    joue_periods_filtrees: Filtered periods where the instrument is played.
    regions: List of regions with their start times.
    instrument_name: The name of the instrument.
    track_name: The name of the track.
    output_dir: The directory where the figure will be saved.
Returns:
    The path to the saved figure.
    """
def plot_rms(rms_values_norm, window_sec, joue_periods_filtrees, regions, instrument_name, track_name, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(rms_values_norm)) * window_sec, rms_values_norm, label='RMS Canal 1 (normalisé)')
    for (start, end) in joue_periods_filtrees:
        plt.axvspan(start, end, color='green', alpha=0.2, label='Instrument joue' if 'Instrument joue' not in plt.gca().get_legend_handles_labels()[1] else "")
    for region in regions:
        name = region["name"]
        color = 'purple' if ("Break" in name or "Punch" in name) else 'red'
        plt.axvline(region["start"], color=color, linestyle='--', alpha=0.7)
        plt.text(region["start"], plt.ylim()[1]*0.95, region["name"][0:3], color=color, ha='left', va='top', fontsize=9, rotation=90)
    plt.xlabel('Temps (s)')
    plt.ylabel('RMS (normalisé)')
    plt.title(f"RMS {instrument_name} - \"{track_name}\" ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rms_{instrument_name}_{track_name}.png")
    plt.savefig(output_path)
    plt.close()
    return output_path

def usage(track_idx, instrument_idx):
    print(f"Usage: python {sys.argv[0]} <track_index> <instrument_index>")
    print(f"Example: python {sys.argv[0]} {track_idx} {instrument_idx}")
    sys.exit(1)


"""
Analyse the audio RMS for a given track and instrument.
Parameters:
    track_idx: The index of the track to analyze.
    instrument_idx: The index of the instrument to analyze.
Returns:
    A dictionary containing the results of the analysis, including paths to saved figures and data.
        """
def analyse_audio_rms(track_idx, instrument_idx):
    if track_idx < 0 or instrument_idx < 0:
        usage(track_idx, instrument_idx)
    result = process_track(
        instrument_idx=instrument_idx,
        selected_num_track=track_idx,
    )
    track_name = result["track_name"]
    instrument_name = result["instrument_name"]
    regions = result["regions"]
    fname = f"../Audio/{track_name}.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, data = wavfile.read(fname)
    channel_idx = instrument_idx + 1  # L'audio comporte le chan "clic"
    data_instr = data[:, channel_idx]
    rms_values, window_sec = compute_rms(data_instr, rate)
    rms_values_norm = normalize_rms(rms_values)
    joue_periods_filtrees, joue_ponctuel, non_joue_periods = detect_periods(rms_values_norm, window_sec)
    output_dir = os.path.join("Results", "rms", track_name)
    output_path = plot_rms(
        rms_values_norm, window_sec, joue_periods_filtrees, regions, instrument_name, track_name, output_dir
    )
    print(f"Figure saved at: {output_path}")
    return {
        "track_name": track_name,
        "instrument_name": instrument_name,
        "output_path": output_path,
        "rms_values": rms_values,
        "rms_values_norm": rms_values_norm,
        "window_sec": window_sec,
        "joue_periods_filtrees": joue_periods_filtrees,
        "joue_ponctuel": joue_ponctuel,
        "non_joue_periods": non_joue_periods,
        "regions": regions,
        "rate": rate,
    }

if __name__ == "__main__":
    analyse_audio_rms(3, 9)
    