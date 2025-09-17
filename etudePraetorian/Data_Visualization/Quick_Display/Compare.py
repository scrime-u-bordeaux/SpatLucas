import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from extract_coords import extract_coords
from RMS import compute_rms

# --- Paramètres à ajuster ---
TRACK_ID = 7           # identifiant du morceau (ex : "id 3-NomDuMorceau;")
INSTR_ID_BASE = 1           # INSTRUMENT SUR LEQUEL ON VA SE BASER
INSTR_ID_TARGETS = [1]  # Liste des identifiants d'instruments cibles potentiels


FS = 5                # fréquence d’échantillonnage spat
START_TIME = 109.0
END_TIME = 118.0
AUDIO_DIR = "Audio"      # dossier contenant les wav

def main():
    # --- 1. Extraire les coordonnées ---
    seq = extract_coords(track_id=TRACK_ID, instr_idx=INSTR_ID_BASE)
    times_spat = seq["times"]
    speeds = seq["speeds"]
    track_name = seq["name"]   # on récupère aussi le nom pour l’audio

    # Norme de la vitesse
    speed_norm = np.linalg.norm(speeds, axis=1)

    # Fenêtrage
    mask_spat = (times_spat >= START_TIME) & (times_spat <= END_TIME)
    times_spat = times_spat[mask_spat]
    speed_norm = speed_norm[mask_spat]

    # --- 2. Charger audio et calculer RMS ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, "../../.."))  # 3 niveaux au-dessus
    audio_path = os.path.join(parent_dir, AUDIO_DIR, f"{track_name}.wav")
    rate, data = wavfile.read(audio_path)

    colors = ["red", "green", "orange", "purple", "brown", "cyan", "magenta"]
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "blue"
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Vitesse spatiale", color=color1)
    ax1.plot(times_spat, speed_norm, label="Vitesse spatiale", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("RMS audio")

    for idx, channel_idx in enumerate(INSTR_ID_TARGETS):
        if data.ndim > 1:
            if channel_idx < data.shape[1]:
                data_instr = data[:, channel_idx]
            else:
                continue  # skip if channel doesn't exist
        else:
            if channel_idx == 0:
                data_instr = data
            else:
                continue  # skip if mono and not channel 0

        rms_values, window_sec = compute_rms(data_instr, rate)
        times_rms = np.arange(len(rms_values)) * window_sec

        mask_rms = (times_rms >= START_TIME) & (times_rms <= END_TIME)
        times_rms_plot = times_rms[mask_rms]
        rms_values_plot = rms_values[mask_rms]

        color2 = colors[idx % len(colors)]
        ax2.plot(times_rms_plot, rms_values_plot, label=f"RMS audio ch {channel_idx}", color=color2)
    
    ax2.tick_params(axis='y')
    ax2.legend(loc="upper right")

    plt.title(f"Comparaison RMS Audio et Vitesse Spatiale - id {TRACK_ID} ({track_name})")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()