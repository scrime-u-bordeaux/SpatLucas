# main.py

import audioRMS
import spat
import sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import Utils

NUM_INSTRUMENTS = 6

def get_num_instruments():
    return NUM_INSTRUMENTS


"""
Usage function to parse command line arguments
Parameters:
    argc: The number of command line arguments.
    argv: The list of command line arguments.
Returns:
    A tuple containing:
    - ok: A boolean indicating if the usage is correct.
    - mode: The mode of operation (audio, spat, mixing).
    - track_idx: The index of the track to analyze.
    - instrument_idx: The index of the instrument to analyze (or None if all instruments).
    - all_instruments: A boolean indicating if all instruments should be analyzed.
"""
def usage(argc, argv):
    if argc < 3 or argv[1] not in ("audio", "spat", "mixing"):
        print("Usage: python visualizer.py <mode> <track_index> [<instrument_index>|all]")
        print("Modes: audio, spat, mixing")
        return False, None, None, None, None, None
    mode = argv[1]
    track_idx = int(argv[2])
    
    instrument_idx = None
    all_instruments = False
    global_mode = False

    if argc == 3:
        global_mode = True
    elif argc >= 4:
        if argv[3] == "all":
            all_instruments = True
        else:
            instrument_idx = int(argv[3])

    return True, mode, track_idx, instrument_idx, all_instruments, global_mode

def main():
    ok, mode, track_idx, instrument_idx, all_instruments, global_mode = usage(len(sys.argv), sys.argv)
    if not ok:
        return

    dataset = []
    
    if global_mode or all_instruments:
        instrument_indices = range(NUM_INSTRUMENTS)
    else:
        instrument_indices = [instrument_idx]

    for idx in instrument_indices:
        entry = {"instrument": idx}
        if mode == "audio":
            entry["audio"] = audioRMS.analyse_audio_rms(track_idx, idx + 1)
        elif mode == "spat":
            entry["spat"] = spat.analyse_spat(track_idx, idx + 1, True, True, False)
        elif mode == "mixing":
            entry["audio"] = audioRMS.analyse_audio_rms(track_idx, idx + 1)
            entry["spat"] = spat.analyse_spat(track_idx, idx + 1, False, True, False)
        dataset.append(entry)

    if global_mode:
        if mode == "audio":
            print("Aucune analyse audio pour le mode global disponible.")
        elif mode == "spat":
            # We could perform a global processing here if needed
            pass
        elif mode == "mixing":
            process_mixing_results(
                [d.get("audio") for d in dataset],
                [d.get("spat") for d in dataset], track_idx, instrument_idx
            )
        return dataset

    if all_instruments:
        if mode == "mixing":
            process_mixing_results(
                [d.get("audio") for d in dataset],
                [d.get("spat") for d in dataset], track_idx, instrument_idx
            )
        # if mode in ("mixing"):
        #     plot_3d_spat_audio(dataset)
        return dataset

    # Case where we have a specific track and instrument
    print(f"Analyse pour le morceau '{Utils.get_track_name(track_idx)}' - instrument '{Utils.get_instrument_name(instrument_idx + 1)}'")
    if mode == "audio":
        return dataset[0].get("audio")
    elif mode == "spat":
        return dataset[0].get("spat")
    elif mode == "mixing":
        return process_mixing_results(
            dataset[0].get("audio"),
            dataset[0].get("spat"), track_idx, instrument_idx
        )

import matplotlib.pyplot as plt


"""
Generates a plot showing the speed of movement over time for a specific track and instrument.
Not functional currently
"""
def process_mixing_results(result_audio, result_spat, track_idx, instrument_idx, line_height=0.8, line_width=0.9):
    """
    Affiche les périodes de jeu et les vitesses de spatialisation par instrument,
    avec les frontières de régions affichées comme de grandes lignes verticales
    traversant toutes les pistes (pas de zones colorées).
    """
    import matplotlib.pyplot as plt

    if not isinstance(result_audio, list):
        result_audio = [result_audio]
        result_spat = [result_spat]

    num_instruments = len(result_audio)
    regions = []
    if result_audio and isinstance(result_audio[0], dict):
        regions = result_audio[0].get("regions", [])

    num_rows = num_instruments
    fig, ax = plt.subplots(
        num_rows, 1,
        figsize=(12, line_height * num_rows),
        sharex=True,
        gridspec_kw={"hspace": 0}  # pas d'espace vertical
    )
    if num_rows == 1:
        ax = [ax]

    # --- Récupération des frontières de régions ---
    boundaries = []
    if regions:
        for i, region in enumerate(regions):
            t_start = region.get("start", None)
            if t_start is not None:
                boundaries.append((t_start, region.get("label", "")))

    # --- Déterminer fin de la timeline (pour la dernière région) ---
    max_time = 0
    for audio, spat in zip(result_audio, result_spat):
        if audio and "joue_periods_filtrees" in audio and audio["joue_periods_filtrees"]:
            max_time = max(max_time, max(period[1] for period in audio["joue_periods_filtrees"]))
        if spat and "real_times" in spat and spat["real_times"]:
            max_time = max(max_time, max(spat["real_times"]))

    # --- Tracer les courbes par instrument ---
    for idx, (audio, spat) in enumerate(zip(result_audio, result_spat)):
        joue_periods = audio.get('joue_periods_filtrees', [])
        real_times = np.array(spat.get('real_times', []))
        vitesse = np.array(spat.get('vitesse', []))
        instrument_label = Utils.get_instrument_name(idx + 1)

        for period in joue_periods:
            t_start, t_end = period
            ax[idx].axvspan(t_start, t_end, color='tab:green', alpha=0.1, ymin=0.1, ymax=0.9)

        if len(vitesse) == len(real_times) - 1:
            vitesse = np.insert(vitesse, 0, 0)
        if len(real_times) > 1 and len(vitesse) == len(real_times):
            ax[idx].plot(real_times, vitesse, color='tab:blue', linewidth=1.0)

        ax[idx].set_ylabel(instrument_label, rotation=0, ha="right", va="center", fontsize=8)
        ax[idx].set_yticks([])

        # --- Ajouter les traits verticaux des régions ---
        for t, label in boundaries:
            ax[idx].axvline(t, color="red", linestyle="--", linewidth=1, alpha=0.7)
            # Option : ajouter le label une seule fois en haut
            if idx == 0 and label:
                ax[idx].text(t, 1.02, label, transform=ax[idx].get_xaxis_transform(),
                             ha="center", va="bottom", rotation=90, fontsize=8, color="red")

    ax[-1].set_xlabel("Temps (s)")
    plt.suptitle("Périodes de jeu, vitesse de spatialisation et frontières de régions")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
