Article : Towards Multi-Brain Decoding in Autism a self-supervised learning approach

- reproduire exactement leur méthode
- EEG, classification, pyTorch
- trouver data/git de l'article, si base de données téléchargeable
- reprendre le code ou recommencer un code à moi
- je dois expliquer à Amel le code de manière technique de telle manière je vais l'implémenter : (exemple = donnees ont ete split en test/train, 
avec train il a cree un cnn puis un deuxieme cnn, cette entree il l'a donne ici etc etc
- voir si mon explication est bonne ou point noir a éclairé
- si pas de bdd on regarde ailleurs
- avancer de ce côté-là comme les données ne sont pas prete
- pas recuperer ce qu'on a car ils sont dans la reproduction et la comprehension de la chose
- chaque chose qu'on a modifié qu'est-ce que ca a donné comme impact

recherches pour données mots cles :
- Brain-to-Brain Communication V2 BBC2 EEG download
- Healthy Brain Network EEG download
- BBC2 hyperscanning EEG Guillaume Dumas
- Towards Multi-Brain Decoding in Autism GitHub
- Ghazaleh Ranjabaran GitHub
- Guillaume Dumas GitHub hyperscanning
- Temporal Shuffling EEG GitHub Banville

bilan réunion amel et planning semaine 4 :
- savoir nombre de données dans BBC2 ?
- label test sur notre propre jeu de données = yeux ouvert/yeux fermé
- amel propose a l'envers commencer par l'étape BBC2 pour notre jeu de données pas HBN = creer reseau de neurone qui dit oui ou non, 
eux ils avaient utiliser 2 pour representer chaque dyade seul + concatenaction + classification. 
- moi je dois faire la meme chose pour yeux ouvert yeux fermées,
- et quand je voudrais changer les données faut que ce soit dans un fichier independant
- car dans lOADER on doit avoir que x et y, mais x et y doivent pouvoir changer quand j'aurais de vrais labels
- possibilité que a partir de ma base de données je genere mon x et mon y
- objectif = commencer a coté
- je dois faire partie de deep learning sur l'extraction de caracteristique c'est a dire, je vais entrainer sur 1000 sujets
amel a dit 'meme si avec 36 ca va marcher on rajoute la partie pre entrainement ca marchera pas, on peut garder la partie pre entrainement
pour l'extraction de features a partir d'un EEG on suit par le machine learning après du coup ca va se jouer selon les resultats et ce qu'on creuse encore
le deep ou on se contente de l'utiliser sur l'autre partie(je sais pas de quelle partie elle parle)"
- on pourra piocher dans d'autres dataset sur des problemes neuro ou autre et on essayera mon algorithme
- du coup je vais tester sur mon 'dataset toy' pour m'entrainer a faire le modele
- elle ne sait pas si on aura de bons resultats avec 36, si bon resultat tant mieux, si on a un 
resultat ameliorable on peut faire de l'augmentation de données, on peut ouvrri sur des sujets un peu plus recherches
- j'ai pas ce csv j'ai que celui la : 
dyade_id,participant_1_raw,participant_2_raw,participant_1_preprocessed,participant_2_preprocessed,label
J1,J1-1MB2012-LMG-edfplus.edf,J1-2LR2012-LMG-edfplus.edf,J1-1MB2012-LMG-edfplus_reconstruct-epo.fif,J1-2LR2012-LMG-edfplus_reconstruct-epo.fif,
J2,J2-1LW1301-LMG-edfplus.edf,J2-2ZL1301-LMG-edfplus.edf,J2-1LW1301-LMG-edfplus_reconstruct-epo.fif,J2-2ZL1301-LMG-edfplus_reconstruct-epo.fif,
J4,J4-1GG1501-GML-edfplus.edf,J4-2EV1501-GML-edfplus.edf,J4-1GG1501-GML-edfplus_reconstruct-epo.fif,J4-2EV1501-GML-edfplus_reconstruct-epo.fif,
J5,J5-1CB1701-MLG-edfplus.edf,J5-2SA1701-MLG-edfplus.edf,J5-1CB1701-MLG-edfplus_reconstruct-epo.fif,J5-2SA1701-MLG-edfplus_reconstruct-epo.fif,
J7,J7-1NB2401-LGM-edfplus.edf,J7-2TB2401-LGM-edfplus.edf,J7-1NB2401-LGM-edfplus_reconstruct-epo.fif,J7-2TB2401-LGM-edfplus_reconstruct-epo.fif,
J8,J8-1JB3101-MGL-edfplus.edf,J8-2CT3101-MGL-edfplus.edf,J8-1JB3101-MGL-edfplus_reconstruct-epo.fif,J8-2CT3101-MGL-edfplus_reconstruct-epo.fif,
J10,J10-AL1705-MLG-edfplus.edf,J10-AS1402-MLG-edfplus.edf,J10-AL1705-MLG-edfplus_reconstruct-epo.fif,J10-AS1402-MLG-edfplus_reconstruct-epo.fif,
J15,J15-1LA0703-GML-edfplus.edf,J15-2ML0703-GML-edfplus.edf,J15-1LA0703-GML-edfplus_reconstruct-epo.fif,J15-2ML0703-GML-edfplus_reconstruct-epo.fif,
