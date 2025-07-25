import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import sys

seq_dir = os.path.join(os.path.dirname(__file__), 'seq')
data = {}
for filename in os.listdir(seq_dir):
    file_path = os.path.join(seq_dir, filename)
    if os.path.isfile(file_path):
        data[filename] = pd.read_csv(file_path, header=None)

print(data.keys())    
sys.exit(0)

###############
select_instrument = 3
###############
# INFO

indexInstrument = {
    0: "Sample",
    1: "Voc 1",
    2: "Voc 2",
    3: "Guitare",
    4: "Basse",
    5: "BatterieG",
    6: "BatterieD",
}


instrument_name = indexInstrument[select_instrument]
filename_seq = f"{instrument_name}.txt"
index_file_instrument = select_instrument
df = data[filename_seq]

figures_groups = []
current_figures = []
track_names = []
track_indices = []

for idx, value in enumerate(df.iloc[:, 1].values):
    value_str = str(value)
    if value_str.strip().startswith("id"):
        if current_figures:
            figures_groups.append(current_figures)
            current_figures = []
        track_part = value_str.split('-', 1)[0].replace('id', '').strip()
        try:
            track_index = int(track_part)
        except ValueError:
            track_index = None
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
track_indices, track_names, figures_groups = zip(*combined)



###############
selectedNumTrack = 1
###############
# INFO

# 1 - APOSTAT
# 2 - ECRAN DE FUMEE
# 3 - L'ENNEMI
# 4 - L'HYPNOSE
# 5 - COMMUNION
# 6 - FACE AUX GEANTS
# 7 - NOUVEAU DIABLE
# 8 - BALLADE ENTRE LES MINES 
# 9 - TEMPS MORT

selected_track_index = selectedNumTrack - 1
figures = figures_groups[selected_track_index]
track_name = track_names[selected_track_index] if len(track_names) > selected_track_index else "Unknown"
result = [
    [float(parts[-3]), float(parts[-2]), float(parts[-1])]
    for value in figures
    if len(parts := value.split()) >= 3
]

######### Plotting the coordinates
x_coords = [coord[1] for coord in result]
y_coords = [coord[2] for coord in result]



# Réechantillonage
import wave
import contextlib
fname = f"Audio/{track_name}.wav"
with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    total_duration = frames / float(rate)
times = [float(value.split()[0]) for value in figures]

# Ecriture des nouvelles coordonnées
with open(f"newSeq/{instrument_name} - {track_name}.txt", 'w', encoding='utf-8') as f:
    prev_time = 0.0
    f.write(f"0 0\n")
    for rel_time_coord in result:
        rel_time = rel_time_coord[0]
        x_coords = rel_time_coord[1]
        y_coords = rel_time_coord[2]
        if(rel_time <= 0 or rel_time >= total_duration): continue
        time = rel_time * total_duration
        # Écrire des valeurs espacées de 0.01 entre prev_time et time
        t = prev_time + 0.01
        while t < time:
            # f.write(f"{t}{x_coords} {y_coords}\n") AFFICHAGE DU TEMPS
            f.write(f"{x_coords} {y_coords}\n")
            t += 0.01
        prev_time = time
        # f.write(f"{time} {x_coords} {y_coords}\n") AFFICHAGE DU TEMPS
        f.write(f"{x_coords} {y_coords}\n")




## ================== AFFICHAGE SPATIALISATION ================== ##

import numpy as np
import matplotlib.pyplot as plt
import re

# Génération du graphique avec couleur dépendant du temps

# On récupère les temps relatifs normalisés (entre 0 et 1) ainsi que les coordonnées x et y
times_norm = [coord[0] for coord in result]
x_coords = [coord[1] for coord in result]
y_coords = [coord[2] for coord in result]

# Fonction pour convertir un timecode en secondes
def timecode_to_seconds(tc):
    match = re.match(r"(\d+):(\d+):(\d+):(\d+)", str(tc))
    if not match:
        return 0.0
    h, m, s, f = map(int, match.groups())
    fps = 25
    return h * 3600 + m * 60 + s + f / fps

csv_audio_path = f"Audio/{track_name}.csv"
regions = []
if os.path.exists(csv_audio_path):
    df_audio_csv = pd.read_csv(csv_audio_path)
    for _, row in df_audio_csv.iterrows():
        start_sec = timecode_to_seconds(row["Start"])
        region = {"name": row["Name"], "start": start_sec}
        regions.append(region)
else:
    print(f"Fichier CSV non trouvé : {csv_audio_path}")

regions = sorted(regions, key=lambda x: x["start"])

region_names = [m["name"] for m in regions[:-1]]
region_timecodes = [(regions[i]["start"], regions[i+1]["start"]) for i in range(len(regions)-1)]
region_colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'cyan', 'olive']

real_times = [coord[0] * total_duration for coord in result]

# Préparer le dossier de sortie
output_dir = os.path.join("Results", "spat", track_name, instrument_name)
os.makedirs(output_dir, exist_ok=True)
positions_dir = os.path.join(output_dir, "positions")
os.makedirs(positions_dir, exist_ok=True)

for i, (region_name, (start_time, end_time)) in enumerate(zip(region_names, region_timecodes)):
    plt.figure(figsize=(5, 5))
    region_indices = [j for j, t in enumerate(real_times) if start_time <= t < end_time]
    region_x = [x_coords[j] for j in region_indices]
    region_y = [y_coords[j] for j in region_indices]
    plt.scatter(region_x, region_y, c=region_colors[i % len(region_colors)], alpha=0.8, edgecolors='k', label=region_name)
    plt.title(f'{instrument_name}\n"{track_name}"\n{region_name}\n{start_time:.2f}s - {end_time:.2f}s')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    plt.tight_layout()
    # Sanitize region_name for filename
    safe_region_name = re.sub(r'[\\/*?:"<>|]', "_", region_name).strip().replace(" ", "_")
    output_path = os.path.join(positions_dir, f"{instrument_name}_{track_name}_{safe_region_name}.png")
    if not os.path.exists(output_path):
        plt.savefig(output_path)
        print(f"Figure enregistrée dans : {output_path}")
    plt.close()

    # Ajout : Heatmap des positions pour cette région
    if region_x and region_y:
        plt.figure(figsize=(5, 5))
        heatmap_size = 10  # résolution de la heatmap (plus petit = plus gros carrés)
        heatmap, xedges, yedges = np.histogram2d(
            region_x, region_y, bins=heatmap_size, range=[[-5, 5], [-5, 5]]
        )
        plt.imshow(
            heatmap.T, cmap='hot', origin='lower',
            extent=[-5, 5, -5, 5], aspect='auto'
        )
        plt.colorbar(label='Densité')
        plt.title(f'Heatmap {instrument_name}\n"{track_name}"\n{region_name}\n{start_time:.2f}s - {end_time:.2f}s')
        plt.xlabel('Coordonnée X')
        plt.ylabel('Coordonnée Y')
        plt.grid(False)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.tight_layout()
        # Nouveau dossier heatmap
        heatmap_dir = os.path.join(output_dir, "heatmap")
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_path = os.path.join(heatmap_dir, f"{instrument_name}_{track_name}_{safe_region_name}_heatmap.png")
        if not os.path.exists(heatmap_path):
            plt.savefig(heatmap_path)
            print(f"Heatmap enregistrée dans : {heatmap_path}")
        plt.close()

# Ajout : Heatmap par instrument pour ce morceau
for idx_instr, instr_name in indexInstrument.items():
    fname_seq = f"{instr_name}.txt"
    if fname_seq in data:
        df_instr = data[fname_seq]
        figures_groups_instr = []
        current_figures_instr = []
        for idx, value in enumerate(df_instr.iloc[:, 1].values):
            value_str = str(value)
            if value_str.strip().startswith("id"):
                if current_figures_instr:
                    figures_groups_instr.append(current_figures_instr)
                    current_figures_instr = []
            else:
                current_figures_instr.append(' '.join(value_str.rstrip(';').strip().split()))
        if current_figures_instr:
            figures_groups_instr.append(current_figures_instr)
        if selected_track_index < len(figures_groups_instr):
            figures_instr = figures_groups_instr[selected_track_index]
            instr_x, instr_y = [], []
            for value in figures_instr:
                parts = value.split()
                if len(parts) >= 3:
                    try:
                        instr_x.append(float(parts[-2]))
                        instr_y.append(float(parts[-1]))
                    except Exception:
                        continue
            if instr_x and instr_y:
                plt.figure(figsize=(6, 6))
                heatmap_size = 20
                heatmap, xedges, yedges = np.histogram2d(
                    instr_x, instr_y, bins=heatmap_size, range=[[-5, 5], [-5, 5]]
                )
                plt.imshow(
                    heatmap.T, cmap='hot', origin='lower',
                    extent=[-5, 5, -5, 5], aspect='auto'
                )
                plt.colorbar(label='Densité')
                plt.title(f'Heatmap {instr_name}\n"{track_name}"')
                plt.xlabel('Coordonnée X')
                plt.ylabel('Coordonnée Y')
                plt.grid(False)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                plt.tight_layout()
                # Dossier heatmap par instrument
                heatmap_dir = os.path.join("Results", "spat", track_name, instr_name, "heatmap")
                os.makedirs(heatmap_dir, exist_ok=True)
                heatmap_path = os.path.join(heatmap_dir, f"{instr_name}_{track_name}_heatmap.png")
                plt.savefig(heatmap_path)
                print(f"Heatmap {instr_name} enregistrée dans : {heatmap_path}")
                plt.close()

# Ajout : Heatmap de tous les instruments pour ce morceau
all_x, all_y = [], []
for idx_instr, instr_name in indexInstrument.items():
    fname_seq = f"{instr_name}.txt"
    if fname_seq in data:
        df_instr = data[fname_seq]
        figures_groups_instr = []
        current_figures_instr = []
        for idx, value in enumerate(df_instr.iloc[:, 1].values):
            value_str = str(value)
            if value_str.strip().startswith("id"):
                if current_figures_instr:
                    figures_groups_instr.append(current_figures_instr)
                    current_figures_instr = []
            else:
                current_figures_instr.append(' '.join(value_str.rstrip(';').strip().split()))
        if current_figures_instr:
            figures_groups_instr.append(current_figures_instr)
        if selected_track_index < len(figures_groups_instr):
            figures_instr = figures_groups_instr[selected_track_index]
            for value in figures_instr:
                parts = value.split()
                if len(parts) >= 3:
                    try:
                        all_x.append(float(parts[-2]))
                        all_y.append(float(parts[-1]))
                    except Exception:
                        continue

if all_x and all_y:
    plt.figure(figsize=(6, 6))
    heatmap_size = 20
    heatmap, xedges, yedges = np.histogram2d(
        all_x, all_y, bins=heatmap_size, range=[[-5, 5], [-5, 5]]
    )
    plt.imshow(
        heatmap.T, cmap='hot', origin='lower',
        extent=[-5, 5, -5, 5], aspect='auto'
    )
    plt.colorbar(label='Densité')
    plt.title(f'Heatmap TOUS INSTRUMENTS\n"{track_name}"')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.grid(False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.tight_layout()
    # Dossier heatmap global
    heatmap_dir = os.path.join("Results", "spat", track_name, "ALL")
    os.makedirs(heatmap_dir, exist_ok=True)
    heatmap_path = os.path.join(heatmap_dir, f"ALL_{track_name}_heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Heatmap TOUT INSTRUMENTS enregistrée dans : {heatmap_path}")
    plt.close()

x_coords_np = np.array(x_coords)
y_coords_np = np.array(y_coords)
real_times_np = np.array(real_times)

dx = np.diff(x_coords_np)
dy = np.diff(y_coords_np)
dt = np.diff(real_times_np)
dt[dt == 0] = 1e-8

vitesse = np.sqrt(dx**2 + dy**2) / dt
temps_milieu = (real_times_np[:-1] + real_times_np[1:]) / 2

output_path = os.path.join(output_dir, f"vitesse_{instrument_name}_{track_name}.png")

plt.figure(figsize=(12, 5))
plt.plot(temps_milieu, vitesse, color='blue', label='Vitesse instantanée')
plt.xlabel('Temps (s)')
plt.ylabel('Vitesse (u/s)')
plt.title(f"Vitesse de déplacement en fonction du temps\n{instrument_name} - \"{track_name}\"")
plt.grid(True)

for i, (region_name, (start_time, end_time)) in enumerate(zip(region_names, region_timecodes)):
    plt.axvspan(start_time, end_time, color=region_colors[i % len(region_colors)], alpha=0.08)
    plt.axvline(start_time, color=region_colors[i % len(region_colors)], linestyle='--', alpha=0.7)
    ylim = plt.ylim()
    y_base = ylim[1] * 0.95
    y_offset = (ylim[1] - ylim[0]) * 0.05
    y_text = y_base - (i % 2) * y_offset
    plt.text((start_time + end_time) / 2, y_text, region_name[0:3], color=region_colors[i % len(region_colors)],
             ha='center', va='top', fontsize=10, alpha=0.8, rotation=0)

plt.tight_layout()
plt.savefig(output_path)
plt.close()


## ================== FIN AFFICHAGE SPATIALISATION ================== ##




## ================== HEATMAP =================== ##


# # Création d'une graine pour la reproductibilité
# np.random.seed(2)

# # Augmenter la résolution de la heatmap
# heatmap_size = 50  # Par exemple, passer de 20 à 50

# # Génération d'un tableau heatmap_size x heatmap_size basé sur les coordonnées x et y
# heatmap, xedges, yedges = np.histogram2d(
#     x_coords, y_coords, bins=heatmap_size, range=[[-5, 5], [-5, 5]]
# )
# data = heatmap.astype(int)

# plt.figure(figsize=(8, 7))
# plt.imshow(data, cmap='Blues', origin='lower', extent=[-5, 5, -5, 5], aspect='auto')

# plt.colorbar()

# # Annotation des valeurs (optionnel, peut être illisible si heatmap_size est grand)
# if heatmap_size <= 30:
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             plt.text(
#                 xedges[j] + (xedges[1] - xedges[0]) / 2,
#                 yedges[i] + (yedges[1] - yedges[0]) / 2,
#                 '%d' % data[i, j],
#                 horizontalalignment='center',
#                 verticalalignment='center',
#                 fontsize=7
#             )

# # Création de listes d'étiquettes de coche (10 ticks pour lisibilité)
# num_ticks = 10
# x_tick_positions = np.linspace(-5, 5, num_ticks)
# y_tick_positions = np.linspace(-5, 5, num_ticks)
# x_labels = [f"{x:.1f}" for x in x_tick_positions]
# y_labels = [f"{y:.1f}" for y in y_tick_positions]

# plt.xticks(x_tick_positions, labels=x_labels)
# plt.yticks(y_tick_positions, labels=y_labels)
# plt.title(f"Heatmap de {instrument_name} - \"{track_name}\"")
# plt.xlabel('Coordonnée X')
# plt.ylabel('Coordonnée Y')
# plt.tight_layout()
# plt.show()


## ================== FIN HEATMAP =================== ##




## ================== ANALYSE AUDIO RMS ================== ##

from scipy.io import wavfile

import warnings
import random
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rate, data = wavfile.read(fname)

# Pour compatibilité avec la suite du code, on garde data0 pour l'instrument sélectionné
channel_idx = index_file_instrument + 1  # L'audio comporte le chan "clic" qui ne nous intéresse pas
data_instr = data[:, channel_idx]

window_sec = 0.01
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
        # On smooth
        alpha = 0.2  # 1 = pas de smoothing, 0 = maximum smoothing
        rms = alpha * rms + (1 - alpha) * rms_values[-1]
    rms_values.append(rms)

# Normalisation des valeurs RMS pour avoir une échelle plus simple (par exemple, entre 0 et 10)
rms_values = np.array(rms_values)
rms_max = np.max(rms_values)
if rms_max > 0:
    rms_values_norm = rms_values / rms_max * 10  # Échelle de 0 à 10
else:
    rms_values_norm = rms_values

# Détection des périodes où l'instrument NE JOUE PAS (seulement si la durée dépasse seuil_temps)

seuil_rms = 1  # seuil sur l'échelle normalisée (0-10)
seuil_temps_non_joue = 2  # durée minimale (en secondes) pour considérer que l'instrument NE joue PAS
seuil_temps_joue = 2       # durée minimale (en secondes) pour considérer que l'instrument JOUE vraiment

joue = rms_values_norm > seuil_rms
non_joue_periods = []
start_idx = None

# Détection des périodes où l'instrument NE JOUE PAS
for idx, val in enumerate(joue):
    if not val and start_idx is None:
        start_idx = idx
    elif val and start_idx is not None:
        duration = (idx - start_idx) * window_sec
        if duration >= seuil_temps_non_joue:
            non_joue_periods.append((start_idx * window_sec, idx * window_sec))
        start_idx = None
# Si la dernière période va jusqu'à la fin
if start_idx is not None:
    duration = (len(joue) - start_idx) * window_sec
    if duration >= seuil_temps_non_joue:
        non_joue_periods.append((start_idx * window_sec, len(joue) * window_sec))

# Calcul des intervalles où ça joue (tout le reste)
joue_periods = []
prev_end = 0.0
for start, end in non_joue_periods:
    if start > prev_end:
        joue_periods.append((prev_end, start))
    prev_end = end
# Ajouter la fin si besoin
if prev_end < len(joue) * window_sec:
    joue_periods.append((prev_end, len(joue) * window_sec))

# Filtrer les périodes où ça joue pour ne garder que celles d'une durée suffisante
joue_periods_filtrees = []
joue_ponctuel = []
for start, end in joue_periods:
    if (end - start) >= seuil_temps_joue:
        joue_periods_filtrees.append((start, end))
    else:
        joue_ponctuel.append((start, end))

plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(rms_values_norm)) * window_sec, rms_values_norm, label='RMS Canal 1 (normalisé)')
for (start, end) in joue_periods_filtrees:
    plt.axvspan(start, end, color='green', alpha=0.2, label='Instrument joue' if 'Instrument joue' not in plt.gca().get_legend_handles_labels()[1] else "")

# Ajout des marqueurs pour chaque région extraite du CSV
for region in regions:
    name = region["name"]
    if "Break" in name or "Punch" in name:
        color = 'purple'
    else:   
        color = 'red'
    plt.axvline(region["start"], color=color, linestyle='--', alpha=0.7)
    plt.text(region["start"], plt.ylim()[1]*0.95, region["name"][0:3], color=color, ha='left', va='top', fontsize=9, rotation=90)

plt.xlabel('Temps (s)')
plt.ylabel('RMS (normalisé)')
plt.title(f"RMS {instrument_name} - \"{track_name}\" ")
plt.grid(True)
plt.legend()
plt.tight_layout()
# Nouveau dossier de sortie : Results/rms/<track_name>/<instrument_name>
output_dir = os.path.join("Results", "rms", track_name)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"rms_{instrument_name}_{track_name}.png")
plt.savefig(output_path)
print(f"Figure enregistrée dans : {output_path}")
plt.close()

## ================== FIN ANALYSE AUDIO RMS ================== ##

# Affichage de la courbe de vitesse avec les périodes où l'instrument joue ou non

# Nouveau dossier de sortie : Results/mix/<track_name>/<instrument_name>
output_dir = os.path.join("Results", "mix", track_name)
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 5))
plt.plot(temps_milieu, vitesse, color='blue', label='Vitesse instantanée')

# Affichage des périodes où l'instrument joue (en vert) et ne joue pas (en rouge pâle)
for (start, end) in joue_periods_filtrees:
    plt.axvspan(start, end, color='green', alpha=0.15, label='Instrument joue' if 'Instrument joue' not in plt.gca().get_legend_handles_labels()[1] else "")
# Affichage des périodes où l'instrument ne joue pas
for (start, end) in non_joue_periods:
    plt.axvspan(start, end, color='red', alpha=0.08, label='Instrument ne joue pas' if 'Instrument ne joue pas' not in plt.gca().get_legend_handles_labels()[1] else "")

for (start, end) in joue_ponctuel:
    plt.axvspan(start, end, color='orange', alpha=0.15, label='Instrument joue ponctuellement' if 'Instrument joue ponctuellement' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel('Temps (s)')
plt.ylabel('Vitesse (u/s)')
plt.title(f"mix spat / rms\n{instrument_name} - \"{track_name}\"")
plt.grid(True)
plt.legend()
plt.tight_layout()
output_path = os.path.join(output_dir, f"mix_{instrument_name}_{track_name}.png")
plt.savefig(output_path)
print(f"Figure enregistrée dans : {output_path}")
plt.close()
