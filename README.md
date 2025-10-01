Stage de fin d'étude, Master 2 informatique, de Lucas Brouet. L’objectif est de générer automatiquement une spatialisation multincanale à partir d'un audio ou d'une partition.
Le cas d'étude dans le cadre de ce stage sont les musiques du dernier album du groupe de rock : "Praetorian" (Furialis). 

Contexte : Une expérience de spatialisation a été réalisée lors d'un concert du groupe à "l'inconnue" (salle de spectacle dans Bordeaux). Durant le concert, un des membre du SCRIME a conçu un système lui permettant de déplacement les sources sonore de chaque instruments dans l'espace. Cela a donné lieu à une spatialisation "intuitive", qui constitue donc une donnée d'entrée pour le stage.
L'objectif sera de se servir de ces données afin de :

- Comprendre les mécaniques et paramètres qui incitent à "déclencher" cette spatialisation.
- Etablir des trajectoires spatiales "types"
- Entrainer un modèle afin de reproduire ce mécanisme à partir des données précedentes. 
- Tester et comparer via du contenu visuel et audio

Le code comporte deux grandes parties :

La **visualisation de donnée** (_Data_Visualization_)ainsi que **l'entrainement des données** (_Data_Training_)


**Visualisation de donnée**
Il s'agit ici de produire du contenu visuelle, comme des graphiques 2d et 3d, afin d'observer et comprendre la spatialisation intuitive. On peut ainsi comprendre les paramètre et moments clés du processus de spat.

**Entrainement de donnée**
Ici on utilise et formate les données afin de réaliser un entrainement pour notre model. L'objectif est de produire une serie de coordonnées pour un instrument. Cette dernière pourra ensuite être exploité sur le logiciel MaxMSP


# Utilisation de l'application
L'intégralité se passe au sein du répertoire "EtudePraetorian"

## Data Training
Pour executer correctement le code, il faut se placer dans le repertoire "Data_Training"

### Génération des dataset
La première étape sera de générer les dataset qui nous permettrons par la suite d'entrainer notre modèle. Vous pouvez executer _get_dataset_ pour cela.

Les séquences que l'on veut générer doivent actuellement être entrées manuellement dans la fonction "main" de get_dataset.py en appelant "build_dataset" (cf. documentation).

Il est possible de ne pas ajouter au dataset certains paramètres en appelant par exemple "include_beats=False" pour le temps. 

### Entrainement du model
Une fois les séquences générées, vous pouvez executer le code python "LSTM.py" afin d'utiliser le model.

L'appel de LSTM va dans un premier temps lire les différentes séquences selectionnées (parmi celles présentes dans "sequences_datasets")qui serviront pour entraîner le modèle.
C'est ensuite que ce dernier produira une suite des coordoonnées pour une ou plusieurs séquences temporelles établies.

Le modèle LSTM utilisé ici apprend à partir de séquences temporelles afin de prédire les coordonnées spatiales (x, y) d’un instrument. Sa complexité dépend du nombre de couches empilées et de la taille cachée (nombre de neurones par couche), ce qui lui permet de mieux capturer la dynamique des données mais augmente aussi le temps d’entraînement.

Les paramètres principaux qui influencent les résultats sont :

numeric_columns : les colonnes de données numériques utilisées comme entrées du modèle, dont les deux dernières doivent correspondre aux coordonnées (x, y) à prédire.

n_epochs : le nombre d’itérations d’entraînement. Plus il est élevé, plus le modèle peut s’adapter aux données, mais avec un risque de surapprentissage.

train_seqs : les séquences utilisées pour entraîner le modèle. La diversité et la quantité de ces données conditionnent la qualité de généralisation.

predict_seqs : les séquences sur lesquelles on applique le modèle pour générer les prédictions de coordonnées.

## Data Visualization
Pour executer correctement le code, il faut se placer dans le repertoire "Data_Visualization"

Possibilité de générer 3 types de données : Audio, spatiales ainsi qu'un mix des deux.

Dans tous les cas, il faudra éxécuter "Visualizer.py" avec les paramètres suivants : <mode> <track_index> [<instrument_index>|all]
Avec 
- mode = "audio", "spat", ou "mixing". 
- track_index = [1-9]
- instrument_index = [0-5] OU "all" dans le cas ou on souhaite analyser tous les instruments en une seule fois (on pourra générer ainsi une heatmap du morceau).

  On peut ensuite consulter les resultats dans Data_Visualization/Results

