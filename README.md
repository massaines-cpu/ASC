# Notes de projet – Stage IA EEG

## Article de référence

**Towards Multi-Brain Decoding in Autism: A Self-Supervised Learning Approach**

### Objectifs

- Reproduire le plus fidèlement possible la méthode proposée dans l'article.
- Comprendre en détail chaque étape de leur pipeline.
- Utiliser **PyTorch** pour l'implémentation.
- Comprendre les différentes parties :
  - préparation des données EEG ;
  - pré-entraînement auto-supervisé (SSL) ;
  - extraction de caractéristiques (*feature extraction*) ;
  - classification.

---

# Recherche des ressources de l'article

## Vérifier l'existence des ressources suivantes

- Jeu de données BBC2 téléchargeable.
- Jeu de données HBN (Healthy Brain Network).
- Code source officiel.
- Dépôt GitHub des auteurs.
- Données complémentaires.

### Si les ressources sont disponibles

- Télécharger les données.
- Télécharger le code.
- Décider si :
  - reprendre leur code ;
  - ou réécrire entièrement une version personnelle afin de mieux comprendre le fonctionnement.

### Si les ressources ne sont pas disponibles

Chercher un autre jeu de données permettant de reproduire la méthode.

---

# Compréhension du pipeline

Je dois être capable d'expliquer le fonctionnement complet du modèle à Amel.

L'objectif n'est pas seulement que le code fonctionne, mais que je comprenne précisément chaque étape.

Par exemple :

1. préparation des données ;
2. séparation Train / Validation / Test ;
3. création des deux CNN ;
4. passage des EEG dans les encodeurs ;
5. génération des embeddings ;
6. concaténation des deux représentations ;
7. classification finale.

Je dois pouvoir expliquer :

- pourquoi chaque étape existe ;
- ce qu'elle reçoit en entrée ;
- ce qu'elle produit en sortie ;
- son rôle dans l'architecture.

---

# Validation de ma compréhension

À chaque étape :

- vérifier que mon explication est correcte ;
- identifier les points que je ne comprends pas encore ;
- demander des éclaircissements si nécessaire.

---

# Reproduction avant adaptation

Ne pas modifier immédiatement la méthode.

Objectif :

1. reproduire la méthode originale ;
2. comprendre pourquoi chaque choix a été fait ;
3. seulement ensuite commencer les adaptations.

Pour chaque modification réalisée, noter :

- ce qui a été changé ;
- pourquoi ce changement a été effectué ;
- son impact sur les résultats.

---

# Recherches à effectuer

## Jeux de données

- Brain-to-Brain Communication V2 BBC2 EEG download
- Healthy Brain Network EEG download
- BBC2 hyperscanning EEG Guillaume Dumas

## Code source

- Towards Multi-Brain Decoding in Autism GitHub
- Ghazaleh Ranjabaran GitHub
- Guillaume Dumas GitHub hyperscanning
- Temporal Shuffling EEG GitHub Banville

---

# Bilan réunion avec Amel – Semaine 4

## Questions à éclaircir

- Connaître le nombre exact de sujets présents dans BBC2.
- Vérifier si les auteurs utilisent exactement les mêmes labels que ceux décrits dans l'article.

---

## Première tâche de classification

Avant de reproduire le pré-entraînement SSL (HBN), commencer directement par la partie **classification** sur notre propre jeu de données.

La première tâche sera :

```text
Yeux ouverts
vs
Yeux fermés
```

L'objectif est de construire le même pipeline de classification que dans l'article.

---

## Architecture à reproduire

Les auteurs utilisent :

```text
EEG A
        \
         Encodeur
          \
           Embedding A
                         \
                          Concaténation
                         /
           Embedding B
          /
         Encodeur
        /
EEG B

        ↓

Classification
```

Dans un premier temps, cette architecture sera adaptée à la tâche :

```text
Yeux ouverts
vs
Yeux fermés
```

---

## Séparation des labels

Les labels ne doivent jamais être intégrés directement dans le code.

Ils devront être générés dans un fichier indépendant.

Le pipeline devra uniquement recevoir :

- `X`
- `y`

L'objectif est de pouvoir changer facilement les labels lorsque de nouvelles tâches de classification seront disponibles.

---

## Génération des données

À partir de la base de données du laboratoire, générer automatiquement :

- `X`
- `y`

afin d'éviter toute dépendance à une tâche particulière.

---

## Début du développement

Commencer dès maintenant la partie Deep Learning, même si le jeu de données est réduit.

Le but est de construire progressivement toute l'architecture.

---

## Pré-entraînement auto-supervisé

Même si les 36 sujets ne suffisent probablement pas pour reproduire correctement la partie SSL, cette étape sera conservée dans le pipeline.

Deux possibilités seront ensuite étudiées :

- utiliser le Deep Learning uniquement pour extraire des caractéristiques (*feature extraction*) avant une classification plus classique ;
- ou entraîner l'ensemble du pipeline en Deep Learning si les résultats sont suffisants.

Le choix dépendra des performances obtenues.

---

## Ouverture vers d'autres jeux de données

Si les performances sont insuffisantes, plusieurs pistes sont envisagées :

- utiliser d'autres bases EEG ;
- tester l'algorithme sur d'autres problématiques en neurosciences ;
- comparer les performances sur plusieurs jeux de données.

---

## Jeu de données "Toy"

Avant d'utiliser les données définitives, utiliser le **dataset toy** pour :

- développer le pipeline ;
- tester le modèle ;
- vérifier le fonctionnement de l'entraînement.

---

## Taille limitée de la base

Le laboratoire ne dispose actuellement que d'environ **36 sujets**.

Plusieurs scénarios sont possibles :

- les résultats sont satisfaisants ;
- les résultats sont insuffisants.

Dans ce second cas, des pistes seront explorées :

- augmentation de données (*Data Augmentation*) ;
- utilisation de bases publiques plus importantes ;
- approfondissement de la partie Deep Learning.

---

## Métadonnées actuellement disponibles

Le fichier actuellement disponible contient uniquement les informations suivantes :

```text
dyade_id
participant_1_raw
participant_2_raw
participant_1_preprocessed
participant_2_preprocessed
label
```

Exemple :

```csv
dyade_id,participant_1_raw,participant_2_raw,participant_1_preprocessed,participant_2_preprocessed,label
J1,J1-1MB2012-LMG-edfplus.edf,J1-2LR2012-LMG-edfplus.edf,J1-1MB2012-LMG-edfplus_reconstruct-epo.fif,J1-2LR2012-LMG-edfplus_reconstruct-epo.fif,
J2,J2-1LW1301-LMG-edfplus.edf,J2-2ZL1301-LMG-edfplus.edf,J2-1LW1301-LMG-edfplus_reconstruct-epo.fif,J2-2ZL1301-LMG-edfplus_reconstruct-epo.fif,
...
```

Les labels devront donc être générés à partir des métadonnées disponibles.

---

# Bilan réunion avec Amel – Semaine 5

## Travail à réaliser

- Commenter l'ensemble du code.
- Expliquer précisément le rôle de chaque fonction.
- Adapter le pipeline pour fonctionner au niveau **participant** plutôt qu'au niveau **dyade** pour la tâche *Yeux ouverts / Yeux fermés*.
- Préparer une version permettant ensuite de revenir facilement au pipeline Multi-Brain.
- Commencer le développement de l'encodeur EEG inspiré de **ShallowConvNet**.
```