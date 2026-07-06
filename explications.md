# Towards Multi-Brain Decoding in Autism a self-supervised learning approach

## Plan 

## **1. Objectif de l’article**

**Problématique** : Peut-on distinguer une interaction entre 2 personnes TD (neurotypique) 
d'une interaction avec au moins une personne ASC (autiste) ?

On a donc 2 classes :

Classe 1 = TD

Classe 2 = ASC

**Idée** : une interaction sociale implique une **synchronisation cérébrale** et cette synchronisation 
pourrait être **différente selon profil** de la personne. L'article propose donc un **modèle capable d'exploiter
directement les signaux EEG** de 2 personnes pour effectuer une classification

Il n'entraîne pas un **CNN** car il y a très peu de données annotées => utilisation du **Self-Supervised Learning (SSL)**

**En quoi consiste le SSL ?**

On donne des signaux EEG non annotés au réseau qui va apprendre une représentation générale de l'EEG.

Avant de lui donner la vraie tâche on lui apprend ce que sont des signaux EEG classiques

Il ne connaît **jamais** :

- TD
- ASC
- synchronisé
- non synchronisé

Il apprend uniquement : **Comment sont organisés les signaux EEG**
Cette représentation va être utilisée pour la classification TD / ASC

## **2. Données utilisées**

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

## **3. Prétraitement**

**Nettoyage des données brutes** :

- interpolation des mauvais canaux
- notch filter = enlever le bruit électrique
- filtre passe bande
- ICA
- découpage époques de 1 s
- suppression des époques bruitées (autoreject)

à la fin on a une matrice de **61 canaux x 501 échantillons**

## **4. Tâche prétexte**

Création d'une fausse tâche inventée pour entraîner le réseau sans utiliser de label
Réseau cherche à résoudre problème artificiel
pourquoi ? parce qu'il y a beaucoup d'EEG sans diagnostic ASC/TC donc on invente une tâche avec une réponse connue
les auteurs utilisent la methode temporal shuffling (TS)

Ils découpent les signaux en époques de 1 sec puis ils construisent des triplets ????
1. époque d'ancrage A
2. époque d'ancrage B
3. époque C

Il doit répondre si l'époque C appartient au même contexte temporel => apprend que les époques proches
dans le temps se ressemblent + que les époques éloignées

## **5. Architecture single-brain**

Premier réseau entrainé en entrée on a : 61 x 501
Les poids sont calculés avec le single-brain

Utilisation de Shallow ConvNet (CNN spécialisé pour EEG) process :

```
model = ShallowConvNet()

train_sur_HBN(model)

torch.save(model.state_dict(), "encoder.pt")
```

encodeur (CNN entier, le réseau entier + poids) =
1. EEG
2. Convolution temporelle : regarde comment le signal évolue dans le temps, détecte alpha, bêta, delta
3. Convolution spatiale : regarde les relations entre électrodes, voir si telle ou telle électrode augmente ou diminue, réseau 
apprend les corrélations spatiales
4. Pooling
5. Activation
6. Embedding : chaque EEG est résumé(informations importantes des participants) par un vecteur

## **6. Architecture multi-brain**

dataset BBC2 

```
model = ShallowConvNet()

model.load_state_dict(torch.load("encoder.pt"))

train_sur_BBC2(model)
```

Dans cette architecture on possède 2 participants A et B
Les 2 EEG vont passer dans 2 encodeurs : EEG A -> encodeur pré-entraîné -> embedding A ; pareil pour EEG B
Ensuite on prend embedding A et embedding B on les concatenatent -> fully connected --> on les classe TD ou ASC

## **7. Baselines**

Baseline 1 :
même architecture mais sans SSL : EEG -> CNN initalisé aléatoirement -> TD ASC, résulat 53% car réseau doit apprendre EEG
et autisme en même temps mais n'y arrive pas

Baseline 2 :
régression logistique pas de DL, calcule des biomarqueurs EEG (puissance alpha, beta etc)
features -> regression logistique -> TD/ASC ; résultats 50% car arrive pas à capturer complexité interactions cérébrales

## **8. Résultats**

1. Prétexte accuracy = 90 % apprend correctement les structures EEG
2. Downstream accuracy = 78 % très proche de l'état de l'art
3. recall = 99%, modèle détecte presque tous les cas ASC
4. AUC

HBN (1000 sujets)

↓

Prétraitement

↓

Découpage en époques

↓

Temporal Shuffling

↓

Shallow ConvNet

↓

Encodeur pré-entraîné
           │
           │
           ▼

BBC2 (dyades)
EEG A ──► Encodeur ──► Embedding A
                            │
EEG B ──► Encodeur ──► Embedding B
                            │
                            ▼
                     Concaténation
                            ▼
                  Couche Fully Connected
                            ▼
                       TD ou ASC
## **9. Points flous**
PHASE 1 : PRÉ-ENTRAÎNEMENT SSL

                  Dataset HBN
          EEG individuels non étiquetés
                         │
                         ▼
              Prétraitement EEG
                         │
                         ▼
              Époques de 1 seconde
                  61 × 501
                         │
                         ▼
        Création de triplets temporels
            Temporal Shuffling
                         │
                         ▼
                  Shallow ConvNet
                    = encodeur
                         │
                         ▼
              Embedding EEG 100D
                         │
                         ▼
        Prédiction : ordre correct ou mélangé
                         │
                         ▼
             Backpropagation sur HBN
                         │
                         ▼
          Poids de l’encodeur sauvegardés


                 PHASE 2 : TÂCHE FINALE BBC2

                  Dataset BBC2
             EEG hyperscanning dyadique
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 EEG participant A                  EEG participant B
    61 × 501                           61 × 501
        │                                 │
        ▼                                 ▼
 Encodeur pré-entraîné           Encodeur pré-entraîné
        │                                 │
        ▼                                 ▼
 Embedding A 100D                 Embedding B 100D
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
              Concaténation
                 200 valeurs
                       │
                       ▼
          Classificateur fully connected
                       │
                       ▼
              Prédiction finale
                  TD ou ASC

## *10. Plan de reproduction**