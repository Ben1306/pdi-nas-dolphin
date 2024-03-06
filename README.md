# Projet NAS/HPO Dolphin

Ce projet a uniquement été testé pour la version de python `Python 3.10`. Il se peut que pour des versions différentes, des conflits de versions des packages utilisés interviennent.

## Description du contenu

Ce projet contient une implémentation du réseau de neurones EfficientNet.
Il permet d'effectuer la modélisation et l'entrainement de différentes variations de ce modèle.
Il permet aussi de lancer un flow de recherche NAS (pour de l'optimisation d'hyperparamètres) pour optimiser quelques contraintes basiques d'un microcontroller en terme de capacité de calcul (taille du modèle et nombre de MACs)

## Imagenette dataset
Télécharger le dataset Imagenette en suivant le lien ci-dessous (version 320px):
https://github.com/fastai/imagenette

### Procédure
Extraire imagenette.zip dans le dossier `/data` et renommer le dossier extrait `imagenette`

## Configuration de l'environnement python

### Créer un environnement virtuel
```
$ python3 -m venv venv
```
Ne pas oublier d'activer l'environnement virtuel. Sur Mac/Linux:
```
$ source venv/bin/activate
```

### Installer les dépendances
```
(venv) $ pip install -r requirements.txt
```

## Scripts disponibles
Plusieurs scripts sont disponibles et chacun d'entre eux produira ses résultats dans un dossier correspondant dans le dossier `outputs/`.

### Obtenir le détail par couches du nombre de paramètres pour des versions allégées d'EfficientNet B0
```
(venv) $ python3 get_models_detail.py
```
Ce script crée un fichier .txt dans le dossier `outputs/models_detail/` qui détaille la composition des variantes du modèle EfficentNet en fonction de ses paramètres d'entrée.

### Entrainer les variations du modèle B0 qui possèdent moins de 1.5M de paramètres
```
(venv) $ python3 train_models.py
```
Ce script lance une procédure d'entrainement des 6 variantes du modèle qui contiennent un nombre de paramètres < 2 Millions.
Il crée ses résultats dans le dossier `outputs/train_b0_models/`.

### Lancer un flow de NAS
La recherche NAS configurée dans ce script porte sur le expand_ratio k du modèle, son nombre de output_channels pour chaque MBBlock, la résolution des images en entrée du modèle, et le ratio de dropout du classifieur.

Avant de lancer le flow, ouvrir le fichier `start_nni_experiment.py`.
Pour valider la condition du nombre maximum de paramètres (ici choisie à 1M), un essai se voit attribuer une accuracy de 0 si le modèle définit par son jeu de paramètres dépasse les 1M de paramètres. Pour cette raison, on ne définit pas de nombre d'essais maximum dans le flow de recherche NAS (car un certain nombre de ces essais sont "nuls") mais on contrôle sa durée via un temps de recherche total maximum.

Pour modifier ce temps maximum, changer le paramètre `experiment.config.max_experiment_duration`.

Le port sur lequel le flow NAS s'exécute est configuré par défaut sur 8001. Si ce port est déjà utilisé, modifier la commande `experiment.run(<port_number>)`
#### /!\ Remarque /!\
Je n'ai pas pu tester que les paramètres `experiment.config.trial_gpu_number = 1`et `experiment.config.training_service.use_active_gpu = True` fonctionnent correctement car je n'ai pas accès à un GPU.

Pour lancer le flow NAS exécuter la commande
```
(venv) $ python3 start_nni_experiment.py
```

Cliquer sur le lien qui s'affiche dans le terminal pour ouvrir l'interface web locale du service qui vient d'être créé.

Pour arrêter l'expérimentation en cours, taper *`Ctrl + C`* dans le terminal courant.

Les résultats de l'expérimentation sont disponibles depuis l'interface web locale créée par NNI

### Exporter les résultats

Une fois que la recherche NAS est terminée, on peut utiliser le CLI fourni par NNI pour exporter les résultats (ou les reporter à la main si ce n'est pas trop fastidieux).

D'abord, afficher la liste des expérimentations
```
(venv) $ nnictl experiment list --all
```

```
(venv)  benji 02:24 ~/Documents/pdi-dolphin-git $ nnictl experiment list --all
--------------------------------------------------------
Experiment information
Id: cmq2e1zp    Name: EfficientNet NAS  Status: RUNNING Port: 8001  Platform: local

--------------------------------------------------------
```
Copier l'identifiant de l'expérimentation correspondante (ici `cmq2e1zp`)
Ensuite créer un dossier pour l'exportation:
```
(venv) $ mkdir outputs/nni_nas
```
Et exécuter la commande:
```
(venv) $ nnictl experiment export [experiment_id] --filename outputs/nni_nas/results.csv --type csv
```
Cela va exporter dans un fichier CSV les paramètres de configurations et l'accuracy correspondante pour tous les essais `Terminés`.
