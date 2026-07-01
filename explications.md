# Towards Multi-Brain Decoding in Autism a self-supervised learning approach

## Plan 

**1. Objectif de l’article**

**Problématique** : Peut-on distinguer une interaction entre 2 personnes TD (neurotypique) 
d'une interaction avec au moins une personne ASC (autiste) ?

On a donc 2 classes :
Classe 1 = TD
Classe 2 = ASC

**Idée** : une interaction sociale implique une **synchronisation cérébrale** et cette synchronisation 
pourrait être **différente selon profil** de la personne. L'article propose donc un **modèle capable d'exploiter
directement les signaux EEG** de 2 personnes pour effectuer une classification

Il n'entraîne pas un **CNN** car il y a très peu de données annotées => utilisation du **Self-Supervised Learning (SSL)**

**En quoi consite le SSL ?**

On donne des signaux EEG non annotés au réseau qui va apprendre une représentation générale de l'EEG.

Avant de lui donner la vraie tâche on lui apprend ce que sont des signaux EEG classiques

Il ne connaît **jamais** :

- TD
- ASC
- synchronisé
- non synchronisé

Il apprend uniquement : **Comment sont organisés les signaux EEG**
Cette représentation va être utilisé pour la classification TD / ASC

**2. Données utilisées**

Ils utilisent 2 jeux de données différents :

**Healthy Brain Network (HBN)** :
- 1000 sujets
- EEG de 61 canaux
- segmentation en époques de 1 seconde

**Brain-to-Brain Communication (BBC2)** :
- 142 120 époques dyadiques
- 61 canaux EEG
- 501 points temporels
- fenêtre de 1 seconde

**3. Prétraitement**

**Nettoyage des données brutes** :

- interpolation des mauvais canaux
- notch filter = enlever le bruit électrique
- filtre passe bande
- ICA
- découpage époques de 1 s
- suppression des époques bruitées (autoreject)

à la fin on a une matrice de **61 canaux x 501 échantillons**

**4. Tâche prétexte**
**5. Architecture single-brain**
**6. Architecture multi-brain**
**7. Baselines**
**8. Résultats**
**9. Points flous**
**10. Plan de reproduction**