Stage de fin d'étude, Master 2 informatique, de Lucas Brouet. L’objectif est de générer une spatialisation multincanale automatiquement à partir d'un audio ou d'une partition.
Le cas d'étude dans le cadre de ce stage sont les musiques du dernier album du groupe de rock : "Praetorian". 

Contexte : Une expérience de spatialisation a été réalisée lors d'un concert du groupe à "l'inconnue" (salle de spectacle dans Bordeaux). Durant le concert, un des membre du SCRIME a conçu un système lui
permettant de déplacement les sources sonore de chaque instruments dans l'espace. Cela a donné lieu à une spatialisation "intuitive", qui constitue donc une donnée d'entrée pour le stage.
L'objectif sera de se servir de ces données afin de :

- Comprendre les mécaniques et paramètres qui incitent à "déclencher" cette spatialisation.
- Etablir des trajectoires spatiales "types"
- Entrainer un modèle afin de reproduire ce mécanisme à partir des données précedentes. 
- Tester et comparer via du contenu visuel et audio

Le code comporte deux grandes parties :

La **visualisation de donnée** (_Data_Visualization_)ainsi que **l'entrainement des données** (_Data_Training_)


**Visualisation de donnée**
Il s'agit ici de produire du contenu visuelle, comme des graphiques 2d et 3d, afin d'observer et comprendre la spatialisation intuitive. On peut ainsi comprendre les paramètre et moments cléfs du processus de spat.

**Entrainement de donnée**
Ici on utilise et formate les données afin de réaliser un entrainement. L'objectif est de produire une serie de coordonnées pour un instrument. Cette dernière pourra ensuite être exploité sur le logiciel MaxMSP


# Utilisation de l'application
L'intégralité se passe au sein du répertoire "EtudePraetorian"
## Data Training
Pour executer correctement le code, il faut se placer dans le repertoire "Data_Training"
### Génération des dataset
La première étape sera de générer les dataset qui nous permettrons par la suite d'entrainer notre modèle. Vous pouvez executer _get_dataset_ pour cela.
Plusieurs paramètres sont modifiables : le TIME_SAMPLE, INSTRUMENT_IDX et TRACK_IDX. Les deux premiers concernent le choix de l'instrument traité ainsi que la chanson. Le troisième propose de modifier l'échantillonage réalisé pour le dataset. (Si c'est 20, on decoupera la chanson toutes les 20ms, et on aura toutes les informations qui nous intéressent associés à ce moment la).

/!\ Pour l'instant (24/07/2025), il n'est pas pertinant de toucher à l'instrument et à l'échantillonage. On ne travail qu'avac la guitare actuellement.

### Entrainement du model
Une fois les models générés, vous pouvez executer le code python "ML.py". Ce dernier va pouvoir s'en saisir pour réaliser un entrainement. Actuellement il utilise tous les datasets pour l'entrainement à l'exception du dernier qui sera testé. (Il est donc préférable que les dataset de toutes les chansons soient générés).
Possibilité de changer le model. C'est un travail en cours, les résultats affichés dans la console seront de l'ordre d'extrême impertience car models inadaptés.

## Data Visualization
Pour executer correctement le code, il faut se placer dans le repertoire "Data_Visualization"

Possibilité de générer 2 types de données : Audio et spatiales

Dans les deux cas, il faudra éxécuter "Visualizer.py" avec les paramètres suivants : <mode> <track_index> [<instrument_index>|all]
Avec 
- mode = "audio", "spat", ou "mixing" (non fonctionnel). 
- track_index = [1-9]
- instrument_index = [0-5] OU "all" dans le cas ou on souhaite analyser tous les instruments en une seule fois

  On peut ensuite consulter les resultats dans Data_Visualization/Results

